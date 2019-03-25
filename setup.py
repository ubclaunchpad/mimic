import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent
# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name='mimic',
    version='0.0.1',
    packages=['mimic', 'mimic.model', 'mimic.tests'],
    url='https://github.com/ubclaunchpad/mimic',
    license='MIT',
    author='ubclaunchpad',
    author_email='',
    description='ML-powered Text Generation Library',
    long_description=README,
    long_description_content_type="text/markdown",
    include_package_data=True
)
