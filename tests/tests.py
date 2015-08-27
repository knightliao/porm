#!/usr/bin/env python
# coding=utf8
import logging
import unittest

from base_models import Blog, Entry, EntryTag, QueryLogHandler, interpolation, quote_char, Membership, Member, Team, \
    Relationship, User
from porm.db.db import logger
from porm.db.models import drop_model_tables, create_model_tables
from porm.execute.query import Node, Q


class BasePeeweeTestCase(unittest.TestCase):
    def setUp(self):
        self.qh = QueryLogHandler()
        logger.setLevel(logging.DEBUG)
        logger.addHandler(self.qh)

    def tearDown(self):
        logger.removeHandler(self.qh)

    def normalize(self, s):
        if interpolation != '?':
            s = s.replace('?', interpolation)
        if quote_char != "`":
            s = s.replace("`", quote_char)
        return s

    def assertQueriesEqual(self, queries):
        queries = [(self.normalize(q), p) for q, p in queries]
        self.assertEqual(queries, self.queries())

    def assertSQLEqual(self, lhs, rhs):
        self.assertEqual(
            self.normalize(lhs[0]),
            self.normalize(rhs[0])
        )
        self.assertEqual(lhs[1], rhs[1])

    def assertSQL(self, query, expected_clauses):
        computed_joins, clauses, alias_map = query.compile_where()
        clauses = [(self.normalize(x), y) for (x, y) in clauses]
        expected_clauses = [(self.normalize(x), y) for (x, y) in expected_clauses]
        self.assertEqual(sorted(clauses), sorted(expected_clauses))

    def assertNodeEqual(self, lhs, rhs):
        self.assertEqual(lhs.connector, rhs.connector)
        self.assertEqual(lhs.negated, rhs.negated)
        for i, lchild in enumerate(lhs.children):
            rchild = rhs.children[i]
            self.assertEqual(type(lchild), type(rchild))
            if isinstance(lchild, Q):
                self.assertEqual(lchild.model, rchild.model)
                self.assertEqual(lchild.query, rchild.query)
            elif isinstance(lchild, Node):
                self.assertNodeEqual(lchild, rchild)
            else:
                raise TypeError("Invalid type passed to assertNodeEqual")


class BaseModelTestCase(BasePeeweeTestCase):
    def setUp(self):
        models = [
            Membership, Member, Team, Relationship,
            User, EntryTag, Entry, Blog
        ]
        drop_model_tables(models, fail_silently=True)
        create_model_tables(models)
        super(BaseModelTestCase, self).setUp()

    def queries(self):
        return [x.msg for x in self.qh.queries]

    def create_blog(self, **kwargs):
        blog = Blog(**kwargs)
        blog.save()
        return blog

    def create_entry(self, **kwargs):
        entry = Entry(**kwargs)
        entry.save()
        return entry

    def create_entry_tag(self, **kwargs):
        entry_tag = EntryTag(**kwargs)
        entry_tag.save()
        return entry_tag