"""
Jacob Dein 2016
nacoustik
Author: Jacob Dein
License: MIT
"""


from setuptools import setup, find_packages

setup(	name='nacoustik',
		version='0.1.0',
		description='Python tools for soundscape wave file analysis',
		author='Jacob Dein',
		author_email='jake@jacobdein.com',
		url='https://github.com/jacobdein/nacoustik',
		packages=find_packages(),
		license='MIT',
		platforms='any',
		classifiers=[
		  'License :: OSI Approved :: MIT License',
		  'Development Status :: 3 - Alpha',
		  'Programming Language :: Python :: 2.7',
		  'Programming Language :: Python :: 3',
		  'Environment :: Console'],
)