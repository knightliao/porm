#!/usr/bin/env python
# coding=utf8
import logging
from logging import config

from base_models import Blog
from porm.execute.query import RawQuery
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