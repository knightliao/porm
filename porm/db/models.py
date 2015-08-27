#!/usr/bin/env python
# coding=utf8
from __future__ import with_statement
import copy

import os
import re
import warnings

from porm.db.db import SqliteDatabase
from porm.db.fields import PrimaryKeyColumn, PrimaryKeyField, ForeignKeyField, Field
from porm.execute.query import SelectQuery, filter_query, DoesNotExist, RawQuery, InsertQuery, DeleteQuery, UpdateQuery


DATABASE_NAME = os.environ.get('PORM_DATABASE', 'porm.db')

# define a default database object in the module scope
database = SqliteDatabase(DATABASE_NAME)


class BaseModelOptions(object):
    indexes = None
    ordering = None
    pk_sequence = None

    def __init__(self, model_class, options=None):
        # configurable options
        options = options or {'database': database}
        for k, v in options.items():
            setattr(self, k, v)

        self.rel_fields = {}
        self.reverse_relations = {}
        self.fields = {}
        self.columns = {}
        self.model_class = model_class

    def prepared(self):
        # called when _meta is finished being initialized
        self.defaults = {}
        for field in self.fields.values():
            if field.default is not None:
                self.defaults[field.name] = field.default

    def get_sorted_fields(self):
        return sorted(self.fields.items(), key=lambda (k, v): (k == self.pk_name and 1 or 2, v._order))

    def get_field_names(self):
        return [f[0] for f in self.get_sorted_fields()]

    def get_fields(self):
        return [f[1] for f in self.get_sorted_fields()]

    def get_field_by_name(self, name):
        if name in self.fields:
            return self.fields[name]
        raise AttributeError('Field named %s not found' % name)

    def get_column_names(self):
        return self.columns.keys()

    def get_column(self, field_or_col):
        if field_or_col in self.fields:
            return self.fields[field_or_col].db_column
        return field_or_col

    def get_related_field_by_name(self, name):
        if name in self.rel_fields:
            return self.fields[self.rel_fields[name]]

    def get_related_field_for_model(self, model, name=None):
        for field in self.fields.values():
            if isinstance(field, ForeignKeyField) and field.to == model:
                if name is None or name == field.name or name == field.db_column:
                    return field

    def get_reverse_related_field_for_model(self, model, name=None):
        for field in model._meta.fields.values():
            if isinstance(field, ForeignKeyField) and field.to == self.model_class:
                if name is None or name == field.name or name == field.db_column:
                    return field

    def get_field_for_related_name(self, model, related_name):
        for field in model._meta.fields.values():
            if isinstance(field, ForeignKeyField) and field.to == self.model_class:
                if field.related_name == related_name:
                    return field

    def rel_exists(self, model):
        return self.get_related_field_for_model(model) or \
               self.get_reverse_related_field_for_model(model)


class BaseModel(type):
    inheritable_options = ['database', 'indexes', 'ordering', 'pk_sequence']

    def __new__(cls, name, bases, attrs):
        cls = super(BaseModel, cls).__new__(cls, name, bases, attrs)

        if not bases:
            return cls

        attr_dict = {}
        meta = attrs.pop('Meta', None)
        if meta:
            attr_dict = meta.__dict__

        for b in bases:
            base_meta = getattr(b, '_meta', None)
            if not base_meta:
                continue

            for (k, v) in base_meta.__dict__.items():
                if k in cls.inheritable_options and k not in attr_dict:
                    attr_dict[k] = v
                elif k == 'fields':
                    for field_name, field_obj in v.items():
                        if isinstance(field_obj, PrimaryKeyField):
                            continue
                        if field_name in cls.__dict__:
                            continue
                        field_copy = copy.deepcopy(field_obj)
                        setattr(cls, field_name, field_copy)

        _meta = BaseModelOptions(cls, attr_dict)

        if not hasattr(_meta, 'db_table'):
            _meta.db_table = re.sub('[^\w]+', '_', cls.__name__.lower())

        if _meta.db_table in _meta.database.adapter.reserved_tables:
            warnings.warn('Table for %s ("%s") is reserved, please override using Meta.db_table' % (
                cls, _meta.db_table,
            ))

        setattr(cls, '_meta', _meta)

        _meta.pk_name = None

        for name, attr in cls.__dict__.items():
            if isinstance(attr, Field):
                attr.add_to_class(cls, name)
                _meta.fields[attr.name] = attr
                _meta.columns[attr.db_column] = attr
                if isinstance(attr, PrimaryKeyField):
                    _meta.pk_name = attr.name

        if _meta.pk_name is None:
            _meta.pk_name = 'id'
            pk = PrimaryKeyField()
            pk.add_to_class(cls, _meta.pk_name)
            _meta.fields[_meta.pk_name] = pk

        _meta.model_name = cls.__name__

        pk_field = _meta.fields[_meta.pk_name]
        pk_col = pk_field.column
        if _meta.pk_sequence and _meta.database.adapter.sequence_support:
            pk_col.attributes['nextval'] = " default nextval('%s')" % _meta.pk_sequence

        _meta.pk_col = pk_field.db_column
        _meta.auto_increment = isinstance(pk_col, PrimaryKeyColumn)

        for field in _meta.fields.values():
            field.class_prepared()

        _meta.prepared()

        if hasattr(cls, '__unicode__'):
            setattr(cls, '__repr__', lambda self: '<%s: %r>' % (
                _meta.model_name, self.__unicode__()))

        exception_class = type('%sDoesNotExist' % _meta.model_name, (DoesNotExist,), {})
        cls.DoesNotExist = exception_class

        return cls


class Model(object):
    __metaclass__ = BaseModel

    def __init__(self, *args, **kwargs):
        self.initialize_defaults()

        for k, v in kwargs.items():
            setattr(self, k, v)

    def initialize_defaults(self):
        for field_name, default in self._meta.defaults.items():
            if callable(default):
                val = default()
            else:
                val = default
            setattr(self, field_name, val)

    def prepared(self):
        # this hook is called when the model has been populated from a db cursor
        pass

    def __eq__(self, other):
        return other.__class__ == self.__class__ and \
               self.get_pk() and \
               other.get_pk() == self.get_pk()

    def __ne__(self, other):
        return other.__class__ == self.__class__ and \
               (self.get_pk() != other.get_pk() or self.get_pk() is None)

    def get_field_dict(self):
        field_dict = {}

        for field in self._meta.fields.values():
            if isinstance(field, ForeignKeyField):
                field_dict[field.name] = getattr(self, field.id_storage)
            else:
                field_dict[field.name] = getattr(self, field.name)

        return field_dict

    @classmethod
    def table_exists(cls):
        return cls._meta.db_table in cls._meta.database.get_tables()

    @classmethod
    def create_table(cls, fail_silently=False, extra=''):
        if fail_silently and cls.table_exists():
            return

        db = cls._meta.database
        db.create_table(cls, extra=extra)

        for field_name, field_obj in cls._meta.fields.items():
            if isinstance(field_obj, ForeignKeyField):
                db.create_foreign_key(cls, field_obj)
            elif field_obj.db_index or field_obj.unique:
                db.create_index(cls, field_obj.name, field_obj.unique)

        if cls._meta.indexes:
            for fields, unique in cls._meta.indexes:
                db.create_index(cls, fields, unique)

    @classmethod
    def drop_table(cls, fail_silently=False):
        cls._meta.database.drop_table(cls, fail_silently)

    @classmethod
    def filter(cls, *args, **kwargs):
        return filter_query(cls, *args, **kwargs)

    @classmethod
    def select(cls, query=None):
        select_query = SelectQuery(cls, query)
        if cls._meta.ordering:
            select_query = select_query.order_by(*cls._meta.ordering)
        return select_query

    @classmethod
    def update(cls, **query):
        return UpdateQuery(cls, **query)

    @classmethod
    def insert(cls, **query):
        return InsertQuery(cls, **query)

    @classmethod
    def delete(cls, **query):
        return DeleteQuery(cls, **query)

    @classmethod
    def raw(cls, sql, *params):
        return RawQuery(cls, sql, *params)

    @classmethod
    def create(cls, **query):
        inst = cls(**query)
        inst.save(force_insert=True)
        return inst

    @classmethod
    def get_or_create(cls, **query):
        try:
            inst = cls.get(**query)
        except cls.DoesNotExist:
            inst = cls.create(**query)
        return inst

    @classmethod
    def get(cls, *args, **kwargs):
        return cls.select().get(*args, **kwargs)

    def get_pk_name(self):
        return self._meta.pk_name

    def get_pk(self):
        return getattr(self, self._meta.pk_name, None)

    def set_pk(self, pk):
        pk_field = self._meta.fields[self._meta.pk_name]
        setattr(self, self._meta.pk_name, pk_field.python_value(pk))

    def get_pk_dict(self):
        return {self.get_pk_name(): self.get_pk()}

    def save(self, force_insert=False):
        field_dict = self.get_field_dict()
        if self.get_pk() and not force_insert:
            field_dict.pop(self._meta.pk_name)
            update = self.update(
                **field_dict
            ).where(**{self._meta.pk_name: self.get_pk()})
            update.execute()
        else:
            if self._meta.auto_increment:
                field_dict.pop(self._meta.pk_name)
            insert = self.insert(**field_dict)
            new_pk = insert.execute()
            if self._meta.auto_increment:
                setattr(self, self._meta.pk_name, new_pk)

    @classmethod
    def collect_models(cls, accum=None):
        # dfs to grab any affected models, then from the bottom up issue
        # proper deletes using subqueries to obtain objects to remove
        accum = accum or []
        models = []

        for related_name, rel_model in cls._meta.reverse_relations.items():
            rel_field = cls._meta.get_field_for_related_name(rel_model, related_name)
            coll = [(rel_model, rel_field.name, rel_field.null)] + accum
            if not rel_field.null:
                models.extend(rel_model.collect_models(coll))

            models.append(coll)
        return models

    def collect_queries(self):
        select_queries = []
        nullable_queries = []
        collected_models = self.collect_models()
        if collected_models:
            for model_joins in collected_models:
                depth = len(model_joins)
                base, last, nullable = model_joins[0]
                query = base.select([base._meta.pk_name])
                for model, join, _ in model_joins[1:]:
                    query = query.join(model, on=last)
                    last = join

                query = query.where(**{last: self.get_pk()})
                if nullable:
                    nullable_queries.append((query, last, depth))
                else:
                    select_queries.append((query, last, depth))
        return select_queries, nullable_queries

    def delete_instance(self, recursive=False):
        # XXX: it is strongly recommended you run this in a transaction if using
        # the recursive delete
        if recursive:
            # reverse relations, i.e. anything that would be orphaned, delete.
            select_queries, nullable_queries = self.collect_queries()

            # currently doesn't work with mysql:
            # http://dev.mysql.com/doc/refman/5.1/en/subquery-restrictions.html
            for query, fk_field, depth in select_queries:
                model = query.model
                if not self._meta.database.adapter.subquery_delete_same_table:
                    query = [obj.get_pk() for obj in query]
                    if not query:
                        continue
                model.delete().where(**{
                    '%s__in' % model._meta.pk_name: query,
                }).execute()
            for query, fk_field, depth in nullable_queries:
                model = query.model
                if not self._meta.database.adapter.subquery_delete_same_table:
                    query = [obj.get_pk() for obj in query]
                    if not query:
                        continue
                model.update(**{fk_field: None}).where(**{
                    '%s__in' % model._meta.pk_name: query,
                }).execute()

        return self.delete().where(**{
            self._meta.pk_name: self.get_pk()
        }).execute()

    def refresh(self, *fields):
        fields = fields or self._meta.get_field_names()
        obj = self.select(fields).get(**{self._meta.pk_name: self.get_pk()})

        for field_name in fields:
            setattr(self, field_name, getattr(obj, field_name))


def find_subclasses(klass, include_self=False):
    accum = []
    for child in klass.__subclasses__():
        accum.extend(find_subclasses(child, True))
    if include_self:
        accum.append(klass)
    return accum


def create_model_tables(models, **create_table_kwargs):
    """Create tables for all given models (in the right order)."""
    for m in sort_models_topologically(models):
        m.create_table(**create_table_kwargs)


def drop_model_tables(models, **drop_table_kwargs):
    """Drop tables for all given models (in the right order)."""
    for m in reversed(sort_models_topologically(models)):
        m.drop_table(**drop_table_kwargs)


def sort_models_topologically(models):
    """Sort models topologically so that parents will precede children."""
    models = set(models)
    seen = set()
    ordering = []

    def dfs(model):
        if model in models and model not in seen:
            seen.add(model)
            for child_model in model._meta.reverse_relations.values():
                dfs(child_model)
            ordering.append(model)  # parent will follow descendants

    # order models by name and table initially to guarantee a total ordering
    names = lambda m: (m._meta.model_name, m._meta.db_table)
    for m in sorted(models, key=names, reverse=True):
        dfs(m)
    return list(reversed(ordering))  # want parents first in output ordering
