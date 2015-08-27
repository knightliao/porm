#!/usr/bin/env python
# coding=utf8
from __future__ import with_statement
import datetime
import decimal

import re

from porm.execute.query import Q
from porm.utils.func import ternary


class Column(object):
    db_field = ''
    template = '%(column_type)s'

    def __init__(self, **attributes):
        self.attributes = self.get_attributes()
        self.attributes.update(**attributes)

    def get_attributes(self):
        return {}

    def python_value(self, value):
        return value

    def db_value(self, value):
        return value

    def render(self, db):
        params = {'column_type': db.column_for_field_type(self.db_field)}
        params.update(self.attributes)
        return self.template % params


class VarCharColumn(Column):
    db_field = 'string'
    template = '%(column_type)s(%(max_length)d)'

    def get_attributes(self):
        return {'max_length': 255}

    def db_value(self, value):
        value = unicode(value or '')
        return value[:self.attributes['max_length']]


class TextColumn(Column):
    db_field = 'text'

    def db_value(self, value):
        return value or ''


def format_date_time(value, formats, post_process=None):
    post_process = post_process or (lambda x: x)
    for fmt in formats:
        try:
            return post_process(datetime.datetime.strptime(value, fmt))
        except ValueError:
            pass
    return value


class DateTimeColumn(Column):
    db_field = 'datetime'

    def get_attributes(self):
        return {
            'formats': [
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
            ]
        }

    def python_value(self, value):
        if isinstance(value, basestring):
            return format_date_time(value, self.attributes['formats'])
        return value


class DateColumn(Column):
    db_field = 'date'

    def get_attributes(self):
        return {
            'formats': [
                '%Y-%m-%d',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
            ]
        }

    def python_value(self, value):
        if isinstance(value, basestring):
            pp = lambda x: x.date()
            return format_date_time(value, self.attributes['formats'], pp)
        elif isinstance(value, datetime.datetime):
            return value.date()
        return value


class TimeColumn(Column):
    db_field = 'time'

    def get_attributes(self):
        return {
            'formats': [
                '%H:%M:%S.%f',
                '%H:%M:%S',
                '%H:%M',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S',
            ]
        }

    def python_value(self, value):
        if isinstance(value, basestring):
            pp = lambda x: x.time()
            return format_date_time(value, self.attributes['formats'], pp)
        elif isinstance(value, datetime.datetime):
            return value.time()
        return value


class IntegerColumn(Column):
    db_field = 'integer'

    def db_value(self, value):
        return value or 0

    def python_value(self, value):
        if value is not None:
            return int(value)


class BigIntegerColumn(IntegerColumn):
    db_field = 'bigint'


class BooleanColumn(Column):
    db_field = 'boolean'

    def db_value(self, value):
        return bool(value)

    def python_value(self, value):
        if value is not None:
            return bool(value)


class FloatColumn(Column):
    db_field = 'float'

    def db_value(self, value):
        return value or 0.0

    def python_value(self, value):
        if value is not None:
            return float(value)


class DoubleColumn(FloatColumn):
    db_field = 'double'


class DecimalColumn(Column):
    db_field = 'decimal'
    template = '%(column_type)s(%(max_digits)d, %(decimal_places)d)'

    def get_attributes(self):
        return {
            'max_digits': 10,
            'decimal_places': 5,
            'auto_round': False,
            'rounding': decimal.DefaultContext.rounding,
        }

    def db_value(self, value):
        D = decimal.Decimal
        if not value:
            return D(0)
        if self.attributes['auto_round']:
            exp = D(10) ** (-self.attributes['decimal_places'])
            return D(str(value)).quantize(exp, rounding=self.attributes['rounding'])
        return value

    def python_value(self, value):
        if value is not None:
            if isinstance(value, decimal.Decimal):
                return value
            return decimal.Decimal(str(value))


class PrimaryKeyColumn(Column):
    db_field = 'primary_key'


class PrimaryKeySequenceColumn(PrimaryKeyColumn):
    db_field = 'primary_key_with_sequence'


class FieldDescriptor(object):
    def __init__(self, field):
        self.field = field
        self._cache_name = '__%s' % self.field.name

    def __get__(self, instance, instance_type=None):
        if instance:
            return getattr(instance, self._cache_name, None)
        return self.field

    def __set__(self, instance, value):
        setattr(instance, self._cache_name, value)


def qdict(op):
    def fn(self, rhs):
        return Q(self.model, **{'%s__%s' % (self.name, op): rhs})

    return fn


class Field(object):
    column_class = None
    default = None
    field_template = "%(column)s%(nullable)s"
    _field_counter = 0
    _order = 0

    def __init__(self, null=False, db_index=False, unique=False, verbose_name=None,
                 help_text=None, db_column=None, default=None, choices=None, *args, **kwargs):
        self.null = null
        self.db_index = db_index
        self.unique = unique
        self.verbose_name = verbose_name
        self.help_text = help_text
        self.db_column = db_column
        self.default = default
        self.choices = choices

        self.attributes = kwargs

        Field._field_counter += 1
        self._order = Field._field_counter

    def add_to_class(self, klass, name):
        self.name = name
        self.model = klass
        self.verbose_name = self.verbose_name or re.sub('_+', ' ', name).title()
        self.db_column = self.db_column or self.name
        self.column = self.get_column()

        setattr(klass, name, FieldDescriptor(self))

    def get_column(self):
        return self.column_class(**self.attributes)

    def render_field_template(self, quote_char=''):
        params = {
            'column': self.column.render(self.model._meta.database),
            'nullable': ternary(self.null, '', ' NOT NULL'),
            'qc': quote_char,
        }
        params.update(self.column.attributes)
        return self.field_template % params

    def db_value(self, value):
        if value is None:
            return None
        return self.column.db_value(value)

    def python_value(self, value):
        return self.column.python_value(value)

    def lookup_value(self, lookup_type, value):
        return self.db_value(value)

    def class_prepared(self):
        pass

    __eq__ = qdict('eq')
    __ne__ = qdict('ne')
    __lt__ = qdict('lt')
    __le__ = qdict('lte')
    __gt__ = qdict('gt')
    __ge__ = qdict('gte')
    __lshift__ = qdict('in')
    __rshift__ = qdict('isnull')
    __mul__ = qdict('contains')
    __pow__ = qdict('icontains')
    __xor__ = qdict('istartswith')

    def __neg__(self):
        return (self.model, self.name, 'DESC')


class CharField(Field):
    column_class = VarCharColumn


class TextField(Field):
    column_class = TextColumn


class DateTimeField(Field):
    column_class = DateTimeColumn


class DateField(Field):
    column_class = DateColumn


class TimeField(Field):
    column_class = TimeColumn


class IntegerField(Field):
    column_class = IntegerColumn


class BigIntegerField(IntegerField):
    column_class = BigIntegerColumn


class BooleanField(IntegerField):
    column_class = BooleanColumn


class FloatField(Field):
    column_class = FloatColumn


class DoubleField(Field):
    column_class = DoubleColumn


class DecimalField(Field):
    column_class = DecimalColumn


class PrimaryKeyField(IntegerField):
    column_class = PrimaryKeyColumn
    field_template = "%(column)s NOT NULL PRIMARY KEY%(nextval)s"

    def __init__(self, column_class=None, *args, **kwargs):
        if kwargs.get('null'):
            raise ValueError('Primary keys cannot be nullable')
        if column_class:
            self.column_class = column_class
        if 'nextval' not in kwargs:
            kwargs['nextval'] = ''
        super(PrimaryKeyField, self).__init__(*args, **kwargs)

    def get_column_class(self):
        # check to see if we're using the default pk column
        if self.column_class == PrimaryKeyColumn:
            # if we have a sequence and can support them, then use the special
            # column class that supports sequences
            if self.model._meta.pk_sequence != None and self.model._meta.database.adapter.sequence_support:
                self.column_class = PrimaryKeySequenceColumn
        return self.column_class

    def get_column(self):
        return self.get_column_class()(**self.attributes)


class ForeignRelatedObject(object):
    def __init__(self, to, field):
        self.to = to
        self.field = field
        self.field_name = self.field.name
        self.field_column = self.field.id_storage
        self.cache_name = '_cache_%s' % self.field_name

    def __get__(self, instance, instance_type=None):
        if not instance:
            return self.field

        if not getattr(instance, self.cache_name, None):
            id = getattr(instance, self.field_column, 0)
            qr = self.to.select().where(**{self.to._meta.pk_name: id})
            try:
                setattr(instance, self.cache_name, qr.get())
            except self.to.DoesNotExist:
                if not self.field.null:
                    raise
        return getattr(instance, self.cache_name, None)

    def __set__(self, instance, obj):
        if self.field.null and obj is None:
            setattr(instance, self.field_column, None)
            setattr(instance, self.cache_name, None)
        else:
            from porm.db.models import Model

            if not isinstance(obj, Model):
                setattr(instance, self.field_column, obj)
            else:
                assert isinstance(obj, self.to), "Cannot assign %s to %s, invalid type" % (obj, self.field.name)
                setattr(instance, self.field_column, obj.get_pk())
                setattr(instance, self.cache_name, obj)


class ReverseForeignRelatedObject(object):
    def __init__(self, related_model, name):
        self.field_name = name
        self.related_model = related_model

    def __get__(self, instance, instance_type=None):
        if not instance:
            raise AttributeError('Reverse relations are only accessibly via instances of the class')

        query = {self.field_name: instance.get_pk()}
        qr = self.related_model.select().where(**query)
        return qr


class ForeignKeyField(IntegerField):
    field_template = '%(column)s%(nullable)s REFERENCES %(qc)s%(to_table)s%(qc)s (%(qc)s%(to_pk)s%(qc)s)%(cascade)s%(extra)s'

    def __init__(self, to, null=False, related_name=None, cascade=False, extra=None, *args, **kwargs):
        self.to = to
        self._related_name = related_name
        self.cascade = cascade
        self.extra = extra

        kwargs.update({
            'cascade': ' ON DELETE CASCADE' if self.cascade else '',
            'extra': self.extra or '',
        })
        super(ForeignKeyField, self).__init__(null=null, *args, **kwargs)

    def add_to_class(self, klass, name):
        self.name = name
        self.model = klass
        self.db_column = self.db_column or self.name + '_id'

        if self.name == self.db_column:
            self.id_storage = self.db_column + '_id'
        else:
            self.id_storage = self.db_column

        if self.to == 'self':
            self.to = self.model

        self.verbose_name = self.verbose_name or re.sub('_', ' ', name).title()

        if self._related_name is not None:
            self.related_name = self._related_name
        else:
            self.related_name = klass._meta.db_table + '_set'

        klass._meta.rel_fields[name] = self.name
        setattr(klass, self.name, ForeignRelatedObject(self.to, self))
        setattr(klass, self.id_storage, None)

        reverse_rel = ReverseForeignRelatedObject(klass, self.name)
        setattr(self.to, self.related_name, reverse_rel)
        self.to._meta.reverse_relations[self.related_name] = klass

    def lookup_value(self, lookup_type, value):
        from porm.db.models import Model

        if isinstance(value, Model):
            return value.get_pk()
        return value or None

    def db_value(self, value):
        from porm.db.models import Model

        if isinstance(value, Model):
            return value.get_pk()
        if self.null and value is None:
            return None
        return self.column.db_value(value)

    def get_column(self):
        to_pk = self.to._meta.get_field_by_name(self.to._meta.pk_name)
        to_col_class = to_pk.get_column_class()
        if to_col_class not in (PrimaryKeyColumn, PrimaryKeySequenceColumn):
            self.column_class = to_pk.get_column_class()
        return self.column_class(**self.attributes)

    def class_prepared(self):
        # unfortunately because we may not know the primary key field
        # at the time this field's add_to_class() method is called, we
        # need to update the attributes after the class has been built
        self.attributes.update({
            'to_table': self.to._meta.db_table,
            'to_pk': self.to._meta.pk_col,
        })
        self.column = self.get_column()

