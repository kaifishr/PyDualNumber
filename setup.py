# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name="PyDualNumber",
    version="1.0.0",
    description="A basic implementation of dual numbers in Python.",
    long_description=readme,
    author="Kai Fischer",
    author_email="kai.fabi@posteo.net",
    url='https://github.com/KaiFabi/PyDualNumber',
    license=license,
    packages=find_packages(exclude=('docs'))
)
