# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import setuptools

with open('README.md', 'r') as f:
    readme = f.read()

with open('LICENSE', 'r') as f:
    license = f.read()

setup(
    name='fiwtools',
    version='0.1.0',
    description='Families In the WIld: A Kinship Recogntion Toolbox.',
    long_description=readme,
    author='Joseph Robinson',
    author_email='robinson.jo@husky.neu.edu',
    url='https://github.com/visionjo/FIW_KRT',
    packages=setuptools.find_packages(),
    license=license,
    # packages=find_packages(exclude=('tests', 'docs'))
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

