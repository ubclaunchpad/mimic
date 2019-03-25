"""Setup file for to configure distribution upload to PyPI."""

import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent
# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name='mimic-text',
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

# How to upload to pypi (Remember to change version number):
# Run these commands in terminal:
#   python setup.py sdist bdist_wheel
#   tar tzf dist/mimic-text-0.0.1.tar.gz    # Checks to see if dist file made
#   twine check dist/*                      # Check if dist can be uploaded
#   twine upload dist/*                     # Uploads to PyPI
