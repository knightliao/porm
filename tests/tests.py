#!/usr/bin/env python
# coding=utf8
import unittest

from base_models import Blog, Entry, EntryTag


class BasePeeweeTestCase(unittest.TestCase):
    def setUp(self):
        Blog.create_table()
        Entry.create_table()
        EntryTag.create_table()

    def tearDown(self):
        EntryTag.drop_table()
        Entry.drop_table()
        Blog.drop_table()

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

