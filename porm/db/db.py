#!/usr/bin/env python
# coding=utf8
import logging
import threading

from porm.utils.func import ternary

from porm.db.adapter import MySQLAdapter, SqliteAdapter


logger = logging.getLogger('peewee.logger')


#
# 事务
#
class transaction(object):
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        self._orig = self.db.get_autocommit()
        self.db.set_autocommit(False)
        self.db.begin()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.db.rollback()
        else:
            self.db.commit()
        self.db.set_autocommit(self._orig)


#
# db
#
class Database(object):
    """
    A high-level api for working with the supported database engines.  `Database`
    provides a wrapper around some of the functions performed by the `Adapter`,
    in addition providing support for:
    - execution of SQL queries
    - creating and dropping tables and indexes
    """

    def require_sequence_support(func):
        def inner(self, *args, **kwargs):
            if not self.adapter.sequence_support:
                raise ValueError('%s adapter does not support sequences' % (self.adapter))
            return func(self, *args, **kwargs)

        return inner

    def __init__(self, adapter, database, threadlocals=False, autocommit=True, **connect_kwargs):
        self.adapter = adapter
        self.init(database, **connect_kwargs)

        if threadlocals:
            self.__local = threading.local()
        else:
            self.__local = type('DummyLocal', (object,), {})

        self._conn_lock = threading.Lock()
        self.autocommit = autocommit

    def init(self, database, **connect_kwargs):
        self.deferred = database is None
        self.database = database
        self.connect_kwargs = connect_kwargs

    def connect(self):
        with self._conn_lock:
            if self.deferred:
                raise Exception('Error, database not properly initialized before opening connection')
            self.__local.conn = self.adapter.connect(self.database, **self.connect_kwargs)
            self.__local.closed = False

    def close(self):
        with self._conn_lock:
            if self.deferred:
                raise Exception('Error, database not properly initialized before closing connection')
            self.adapter.close(self.__local.conn)
            self.__local.closed = True

    def get_conn(self):
        if not hasattr(self.__local, 'closed') or self.__local.closed:
            self.connect()
        return self.__local.conn

    def is_closed(self):
        return getattr(self.__local, 'closed', True)

    def get_cursor(self):
        return self.get_conn().cursor()

    def execute(self, sql, params=None, require_commit=True):
        cursor = self.get_cursor()
        res = cursor.execute(sql, params or ())
        if require_commit and self.get_autocommit():
            self.commit()
        logger.debug((sql, params))
        return cursor

    def begin(self):
        pass

    def commit(self):
        self.get_conn().commit()

    def rollback(self):
        self.get_conn().rollback()

    def set_autocommit(self, autocommit):
        self.__local.autocommit = autocommit

    def get_autocommit(self):
        if not hasattr(self.__local, 'autocommit'):
            self.set_autocommit(self.autocommit)
        return self.__local.autocommit

    def commit_on_success(self, func):
        def inner(*args, **kwargs):
            orig = self.get_autocommit()
            self.set_autocommit(False)
            self.begin()
            try:
                res = func(*args, **kwargs)
                self.commit()
            except:
                self.rollback()
                raise
            else:
                return res
            finally:
                self.set_autocommit(orig)

        return inner

    def last_insert_id(self, cursor, model):
        if model._meta.auto_increment:
            return self.adapter.last_insert_id(cursor, model)

    def rows_affected(self, cursor):
        return self.adapter.rows_affected(cursor)

    def quote_name(self, name):
        return ''.join((self.adapter.quote_char, name, self.adapter.quote_char))

    def column_for_field(self, field):
        return self.column_for_field_type(field.get_db_field())

    def column_for_field_type(self, db_field_type):
        try:
            return self.adapter.get_field_types()[db_field_type]
        except KeyError:
            raise AttributeError('Unknown field type: "%s", valid types are: %s' % \
                                 db_field_type, ', '.join(self.adapter.get_field_types().keys())
            )

    def field_sql(self, field):
        rendered = field.render_field_template(self.adapter.quote_char)
        return '%s %s' % (self.quote_name(field.db_column), rendered)

    def get_column_sql(self, model_class):
        return map(self.field_sql, model_class._meta.get_fields())

    def create_table_query(self, model_class, safe, extra='', framing=None):
        if model_class._meta.pk_sequence and self.adapter.sequence_support:
            if not self.sequence_exists(model_class._meta.pk_sequence):
                self.create_sequence(model_class._meta.pk_sequence)
        framing = framing or 'CREATE TABLE %s%s (%s)%s;'
        safe_str = safe and 'IF NOT EXISTS ' or ''
        columns = self.get_column_sql(model_class)

        if extra:
            extra = ' ' + extra

        table = self.quote_name(model_class._meta.db_table)

        return framing % (safe_str, table, ', '.join(columns), extra)

    def create_table(self, model_class, safe=False, extra=''):
        self.execute(self.create_table_query(model_class, safe, extra))

    def create_index_query(self, model_class, field_names, unique, framing=None):
        framing = framing or 'CREATE %(unique)s INDEX %(index)s ON %(table)s(%(field)s);'

        if isinstance(field_names, basestring):
            field_names = (field_names,)

        columns = []
        for field_name in field_names:
            if field_name not in model_class._meta.fields:
                raise AttributeError(
                    'Field %s not on model %s' % (field_name, model_class)
                )
            else:
                field_obj = model_class._meta.fields[field_name]
                columns.append(field_obj.db_column)

        db_table = model_class._meta.db_table
        index_name = self.quote_name('%s_%s' % (db_table, '_'.join(columns)))

        unique_expr = ternary(unique, 'UNIQUE', '')

        return framing % {
            'unique': unique_expr,
            'index': index_name,
            'table': self.quote_name(db_table),
            'field': ','.join(map(self.quote_name, columns)),
        }

    def create_index(self, model_class, field_names, unique=False):
        self.execute(self.create_index_query(model_class, field_names, unique))

    def create_foreign_key(self, model_class, field):
        return self.create_index(model_class, field.name, field.unique)

    def drop_table(self, model_class, fail_silently=False):
        framing = fail_silently and 'DROP TABLE IF EXISTS %s;' or 'DROP TABLE %s;'
        self.execute(framing % self.quote_name(model_class._meta.db_table))

    def add_column_sql(self, model_class, field_name):
        field = model_class._meta.fields[field_name]
        return 'ALTER TABLE %s ADD COLUMN %s' % (
            self.quote_name(model_class._meta.db_table),
            self.field_sql(field),
        )

    def rename_column_sql(self, model_class, field_name, new_name):
        # this assumes that the field on the model points to the *old* fieldname
        field = model_class._meta.fields[field_name]
        return 'ALTER TABLE %s RENAME COLUMN %s TO %s' % (
            self.quote_name(model_class._meta.db_table),
            self.quote_name(field.db_column),
            self.quote_name(new_name),
        )

    def drop_column_sql(self, model_class, field_name):
        field = model_class._meta.fields[field_name]
        return 'ALTER TABLE %s DROP COLUMN %s' % (
            self.quote_name(model_class._meta.db_table),
            self.quote_name(field.db_column),
        )

    @require_sequence_support
    def create_sequence(self, sequence_name):
        return self.execute('CREATE SEQUENCE %s;' % self.quote_name(sequence_name))

    @require_sequence_support
    def drop_sequence(self, sequence_name):
        return self.execute('DROP SEQUENCE %s;' % self.quote_name(sequence_name))

    def get_indexes_for_table(self, table):
        raise NotImplementedError

    def get_tables(self):
        raise NotImplementedError

    def sequence_exists(self, sequence):
        raise NotImplementedError

    def transaction(self):
        return transaction(self)


#
# sqlite
#
class SqliteDatabase(Database):
    def __init__(self, database, **connect_kwargs):
        super(SqliteDatabase, self).__init__(SqliteAdapter(), database, **connect_kwargs)

    def get_indexes_for_table(self, table):
        res = self.execute('PRAGMA index_list(%s);' % self.quote_name(table))
        rows = sorted([(r[1], r[2] == 1) for r in res.fetchall()])
        return rows

    def get_tables(self):
        res = self.execute('select name from sqlite_master where type="table" order by name')
        return [r[0] for r in res.fetchall()]

    def drop_column_sql(self, model_class, field_name):
        raise NotImplementedError('Sqlite3 does not have direct support for dropping columns')

    def rename_column_sql(self, model_class, field_name, new_name):
        raise NotImplementedError('Sqlite3 does not have direct support for renaming columns')


#
# mysql
#
class MySQLDatabase(Database):
    def __init__(self, database, **connect_kwargs):
        super(MySQLDatabase, self).__init__(MySQLAdapter(), database, **connect_kwargs)

    def create_foreign_key(self, model_class, field):
        framing = """
            ALTER TABLE %(table)s ADD CONSTRAINT %(constraint)s
            FOREIGN KEY (%(field)s) REFERENCES %(to)s(%(to_field)s)%(cascade)s;
        """
        db_table = model_class._meta.db_table
        constraint = 'fk_%s_%s_%s' % (
            db_table,
            field.to._meta.db_table,
            field.db_column,
        )

        query = framing % {
            'table': self.quote_name(db_table),
            'constraint': self.quote_name(constraint),
            'field': self.quote_name(field.db_column),
            'to': self.quote_name(field.to._meta.db_table),
            'to_field': self.quote_name(field.to._meta.pk_col),
            'cascade': ' ON DELETE CASCADE' if field.cascade else '',
        }

        self.execute(query)
        return super(MySQLDatabase, self).create_foreign_key(model_class, field)

    def rename_column_sql(self, model_class, field_name, new_name):
        field = model_class._meta.fields[field_name]
        return 'ALTER TABLE %s CHANGE COLUMN %s %s %s' % (
            self.quote_name(model_class._meta.db_table),
            self.quote_name(field.db_column),
            self.quote_name(new_name),
            field.render_field_template(self.adapter.quote_char),
        )

    def get_indexes_for_table(self, table):
        res = self.execute('SHOW INDEXES IN %s;' % self.quote_name(table))
        rows = sorted([(r[2], r[1] == 0) for r in res.fetchall()])
        return rows

    def get_tables(self):
        res = self.execute('SHOW TABLES;')
        return [r[0] for r in res.fetchall()]
