from setuptools import find_packages, setup

setup(
    name='piecewise',
    version='0.1',
    description='Piecewise linear regression',
    url='http://github.com/datadog/piecewise',
    author='Stephen Kappel',
    author_email='stephen@datadoghq.com',
    license='BSD-3-Clause',
    packages=['piecewise'],
    install_requires=[
        'numpy>=1.10.0'
    ],
    extras_require = {
        'plotting':  ['matplotlib>=1.4.3'],
    }
)
