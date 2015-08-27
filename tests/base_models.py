# encoding=utf-8

from __future__ import with_statement
import logging
import datetime

import os

from porm.db.db import MySQLDatabase, SqliteDatabase
from porm.db.fields import CharField, PrimaryKeyField, TextField, ForeignKeyField, BooleanField, DateTimeField
from porm.db.models import Model


class QueryLogHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        self.queries = []
        logging.Handler.__init__(self, *args, **kwargs)

    def emit(self, record):
        self.queries.append(record)


BACKEND = os.environ.get('PEEWEE_TEST_BACKEND', 'sqlite')
TEST_VERBOSITY = int(os.environ.get('PEEWEE_TEST_VERBOSITY') or 1)

database_params = {}

if BACKEND == 'mysql':
    database_class = MySQLDatabase
    database_name = 'peewee_test'
else:
    database_class = SqliteDatabase
    database_name = 'tmp.db'
    import sqlite3

    print 'SQLITE VERSION: %s' % sqlite3.version

test_db = database_class(database_name, **database_params)
interpolation = test_db.adapter.interpolation
quote_char = test_db.adapter.quote_char


class TestModel(Model):
    class Meta:
        database = test_db


# test models
class Blog(TestModel):
    title = CharField()

    def __unicode__(self):
        return self.title


class Entry(TestModel):
    pk = PrimaryKeyField()
    title = CharField(max_length=50, verbose_name='Wacky title')
    content = TextField(default='')
    pub_date = DateTimeField(null=True)
    blog = ForeignKeyField(Blog, cascade=True)

    def __unicode__(self):
        return '%s: %s' % (self.blog.title, self.title)

    def __init__(self, *args, **kwargs):
        self._prepared = False
        super(Entry, self).__init__(*args, **kwargs)

    def prepared(self):
        self._prepared = True


class EntryTag(TestModel):
    tag = CharField(max_length=50)
    entry = ForeignKeyField(Entry)

    def __unicode__(self):
        return self.tag


class EntryTwo(Entry):
    title = TextField()
    extra_field = CharField()


class User(TestModel):
    username = CharField(max_length=50)
    blog = ForeignKeyField(Blog, null=True)
    active = BooleanField(db_index=True, default=False)

    class Meta:
        db_table = 'users'

    def __unicode__(self):
        return self.username


class Relationship(TestModel):
    from_user = ForeignKeyField(User, related_name='relationships')
    to_user = ForeignKeyField(User, related_name='related_to')


class Team(TestModel):
    name = CharField()


class Member(TestModel):
    username = CharField()


class Membership(TestModel):
    team = ForeignKeyField(Team)
    member = ForeignKeyField(Member)


class DefaultVals(TestModel):
    published = BooleanField(default=True)
    pub_date = DateTimeField(default=datetime.datetime.now, null=True)
