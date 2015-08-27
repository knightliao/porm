#!/usr/bin/env python
# coding=utf8
import logging


logger = logging.getLogger('peewee.logger')


class ImproperlyConfigured(Exception):
    pass