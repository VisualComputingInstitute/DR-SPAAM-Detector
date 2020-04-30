from setuptools import setup, find_packages

setup(
    name="dr_spaam",
    version="1.0",
    author='Dan Jia',
    author_email='jia@vision.rwth-aachen.de',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    license='LICENSE.txt',
    description='DR-SPAAM, a deep-learning based person detector for 2D range data.'
)
