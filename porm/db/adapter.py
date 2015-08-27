#!/usr/bin/env python
# coding=utf8
from __future__ import with_statement
import datetime
import decimal

from porm.exception.errors import ImproperlyConfigured


try:
    import sqlite3
except ImportError:
    sqlite3 = None

try:
    import MySQLdb as mysql
except ImportError:
    mysql = None

__all__ = [
    'ImproperlyConfigured', 'SqliteDatabase', 'MySQLDatabase', 'PostgresqlDatabase',
    'asc', 'desc', 'Count', 'Max', 'Min', 'Sum', 'Avg', 'Q', 'Field', 'CharField', 'TextField',
    'DateTimeField', 'BooleanField', 'DecimalField', 'FloatField', 'IntegerField',
    'PrimaryKeyField', 'ForeignKeyField', 'DoubleField', 'BigIntegerField', 'Model',
    'filter_query', 'annotate_query', 'F', 'R', 'DateField', 'TimeField',
    'transaction',
]

if sqlite3 is None and mysql is None:
    raise ImproperlyConfigured('Either sqlite3, or MySQLdb must be installed')

if sqlite3:
    sqlite3.register_adapter(decimal.Decimal, str)
    sqlite3.register_adapter(datetime.date, str)
    sqlite3.register_adapter(datetime.time, str)
    sqlite3.register_converter('decimal', lambda v: decimal.Decimal(v))


#
# base
#
class BaseAdapter(object):
    """
    The various subclasses of `BaseAdapter` provide a bridge between the high-
    level `Database` abstraction and the underlying python libraries like
    psycopg2.  It also provides a way to unify the pythonic field types with
    the underlying column types used by the database engine.

    The `BaseAdapter` provides two types of mappings:
    - mapping between filter operations and their database equivalents
    - mapping between basic field types and their database column types

    The `BaseAdapter` also is the mechanism used by the `Database` class to:
    - handle connections with the database
    - extract information from the database cursor
    """
    operations = {'eq': '= %s'}
    interpolation = '%s'
    sequence_support = False
    for_update_support = False
    subquery_delete_same_table = True
    reserved_tables = []
    quote_char = '"'

    def get_field_types(self):
        field_types = {
            'integer': 'INTEGER',
            'bigint': 'INTEGER',
            'float': 'REAL',
            'decimal': 'DECIMAL',
            'double': 'REAL',
            'string': 'VARCHAR',
            'text': 'TEXT',
            'datetime': 'DATETIME',
            'time': 'TIME',
            'date': 'DATE',
            'primary_key': 'INTEGER',
            'primary_key_with_sequence': 'INTEGER',
            'foreign_key': 'INTEGER',
            'boolean': 'SMALLINT',
            'blob': 'BLOB',
        }
        field_types.update(self.get_field_overrides())
        return field_types

    def get_field_overrides(self):
        return {}

    def connect(self, database, **kwargs):
        raise NotImplementedError

    def close(self, conn):
        conn.close()

    def op_override(self, field, op, value):
        return op

    def lookup_cast(self, field, lookup, value):
        """
        When a lookup is being performed as a part of a WHERE clause, provides
        a way to alter the incoming value that is passed to the database driver
        as part of the list of parameters
        """
        if lookup in ('contains', 'icontains'):
            return '%%%s%%' % value
        elif lookup in ('startswith', 'istartswith'):
            return '%s%%' % value
        return value

    def last_insert_id(self, cursor, model):
        return cursor.lastrowid

    def rows_affected(self, cursor):
        return cursor.rowcount


#
# sql lite
#
class SqliteAdapter(BaseAdapter):
    # note the sqlite library uses a non-standard interpolation string
    operations = {
        'lt': '< %s',
        'lte': '<= %s',
        'gt': '> %s',
        'gte': '>= %s',
        'eq': '= %s',
        'ne': '!= %s',  # watch yourself with this one
        'in': 'IN (%s)',  # special-case to list q-marks
        'is': 'IS %s',
        'isnull': 'IS NULL',
        'between': 'BETWEEN %s AND %s',
        'ieq': "LIKE %s ESCAPE '\\'",  # case-insensitive equality
        'icontains': "LIKE %s ESCAPE '\\'",  # surround param with %'s
        'contains': "GLOB %s",  # surround param with *'s
        'istartswith': "LIKE %s ESCAPE '\\'",
        'startswith': "GLOB %s",
    }
    interpolation = '?'

    def connect(self, database, **kwargs):
        if not sqlite3:
            raise ImproperlyConfigured('sqlite3 must be installed on the system')
        return sqlite3.connect(database, **kwargs)

    def lookup_cast(self, field, lookup, value):
        if lookup == 'contains':
            return '*%s*' % value
        elif lookup == 'icontains':
            return '%%%s%%' % value
        elif lookup == 'startswith':
            return '%s*' % value
        elif lookup == 'istartswith':
            return '%s%%' % value
        return value


#
# mysql
#
class MySQLAdapter(BaseAdapter):
    operations = {
        'lt': '< %s',
        'lte': '<= %s',
        'gt': '> %s',
        'gte': '>= %s',
        'eq': '= %s',
        'ne': '!= %s',  # watch yourself with this one
        'in': 'IN (%s)',  # special-case to list q-marks
        'is': 'IS %s',
        'isnull': 'IS NULL',
        'between': 'BETWEEN %s AND %s',
        'ieq': 'LIKE %s',  # case-insensitive equality
        'icontains': 'LIKE %s',  # surround param with %'s
        'contains': 'LIKE BINARY %s',  # surround param with *'s
        'istartswith': 'LIKE %s',
        'startswith': 'LIKE BINARY %s',
    }
    quote_char = '`'
    for_update_support = True
    subquery_delete_same_table = False

    def connect(self, database, **kwargs):
        if not mysql:
            raise ImproperlyConfigured('MySQLdb must be installed on the system')
        conn_kwargs = {
            'charset': 'utf8',
            'use_unicode': True,
        }
        conn_kwargs.update(kwargs)
        return mysql.connect(db=database, **conn_kwargs)

    def get_field_overrides(self):
        return {
            'primary_key': 'INTEGER AUTO_INCREMENT',
            'boolean': 'bool',
            'float': 'float',
            'double': 'double precision',
            'bigint': 'bigint',
            'text': 'longtext',
            'decimal': 'numeric',
        }
