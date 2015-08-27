import os
from setuptools import setup, find_packages

f = open(os.path.join(os.path.dirname(__file__), 'README.md'))
readme = f.read()
f.close()

setup(
    name='porm',
    version="0.1.0",
    description='a simple python orm',
    long_description=readme,
    author='knightliao',
    author_email='knightliao@gmail.com',
    url='http://github.com/knightliao/porm',
    packages=find_packages(exclude=['tests', 'tests.*']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ],
    test_suite='tests.runtests.runtests'
)
