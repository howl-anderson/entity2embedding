from distutils.core import setup
import sys

PY2 = sys.version_info[0] == 2

required_packages = [
    'tensorflow',
    'sklearn',
    'scipy'
]

if PY2:
    required_packages.append("typing")

setup(
    name='entity2embedding',
    version='0.1.1',
    packages=['entity2embedding'],
    url='',
    license='LGPL',
    author='Xiaoquan Kong',
    author_email='u1mail2m@gmail.com',
    description='Package for word2vec in tensorflow and more',
    requires=required_packages
)
