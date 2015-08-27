#!/usr/bin/env python
# coding=utf8
import logging
from logging import config

from base_models import Blog
from porm.execute.query import RawQuery, SelectQuery
from base_tests import BasePormTestCase


config.fileConfig("log/log.conf.release", defaults={})
root = logging.getLogger()


class QueryTests(BasePormTestCase):
    def test_raw(self):
        rq = RawQuery(Blog, 'SELECT id, title FROM blog')
        logging.info(rq.sql())
        self.assertSQLEqual(rq.sql(), ('SELECT id, title FROM blog', []))

        rq = RawQuery(Blog, 'SELECT id, title FROM blog WHERE title = ?', 'a')
        logging.info(rq.sql())
        self.assertSQLEqual(rq.sql(), ('SELECT id, title FROM blog WHERE title = ?', ['a']))

        rq = RawQuery(Blog, 'SELECT id, title FROM blog WHERE title = ? OR title = ?', 'a', 'b')
        logging.info(rq.sql())
        self.assertSQLEqual(rq.sql(), ('SELECT id, title FROM blog WHERE title = ? OR title = ?', ['a', 'b']))

    def test_select(self):
        sq = SelectQuery(Blog, '*')
        logging.info(sq.sql())
        self.assertSQLEqual(sq.sql(), ('SELECT `id`, `title` FROM `blog`', []))

        sq = SelectQuery(Blog, '*').where(title='a')
        logging.info(sq.sql())
        self.assertSQLEqual(sq.sql(), ('SELECT `id`, `title` FROM `blog` WHERE `title` = ?', ['a']))
        sq2 = SelectQuery(Blog, '*').where(Blog.title == 'a')
        logging.info(sq.sql())
        self.assertEqual(sq.sql(), sq2.sql())

        sq = SelectQuery(Blog, '*').where(title='a', id=1)
        logging.info(sq.sql())
        self.assertSQLEqual(sq.sql(), ('SELECT `id`, `title` FROM `blog` WHERE (`id` = ? AND `title` = ?)', [1, 'a']))
        sq2 = SelectQuery(Blog, '*').where((Blog.id == 1) & (Blog.title == 'a'))
        logging.info(sq.sql())
        self.assertEqual(sq.sql(), sq2.sql())

        # check that chaining works as expected
        sq = SelectQuery(Blog, '*').where(title='a').where(id=1)
        logging.info(sq.sql())
        self.assertSQLEqual(sq.sql(), ('SELECT `id`, `title` FROM `blog` WHERE `title` = ? AND `id` = ?', ['a', 1]))
        sq2 = SelectQuery(Blog, '*').where(Blog.title == 'a').where(Blog.id == 1)
        logging.info(sq.sql())
        self.assertEqual(sq.sql(), sq2.sql())

        # check that IN query special-case works
        sq = SelectQuery(Blog, '*').where(title__in=['a', 'b'])
        logging.info(sq.sql())
        self.assertSQLEqual(sq.sql(), ('SELECT `id`, `title` FROM `blog` WHERE `title` IN (?,?)', ['a', 'b']))
        sq2 = SelectQuery(Blog, '*').where(Blog.title << ['a', 'b'])
        logging.info(sq.sql())
        self.assertEqual(sq.sql(), sq2.sql())