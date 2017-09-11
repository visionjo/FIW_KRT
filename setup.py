# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='FIW_KRT',
    version='0.1.0',
    description='Families In the WIld: A Kinship Recogntion Toolbox.',
    long_description=readme,
    author='Kenneth Reitz',
    author_email='robinson.jo@husky.neu.edu',
    url='https://github.com/huskyjo/FIW_KRT',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

