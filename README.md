# 💬 Mimic

[![Build Status](https://travis-ci.com/ubclaunchpad/mimic.svg?branch=master)](https://travis-ci.com/ubclaunchpad/mimic)

Mimic mimics the style of a given corpus, for great fun and delight.

## Developer Installation

We use [pipenv](https://pipenv.readthedocs.io/en/latest/) for dependency management.

```bash
git clone https://github.com/ubclaunchpad/mimic.git
cd mimic/
pip install pipenv
pipenv install --dev
```

`pipenv` will manage a [virtualenv](https://virtualenv.pypa.io/en/stable/),
so interacting with the program or using the development tools has to be done
through pipenv, like so:

```bash
pipenv run pycodestyle .
```

This can get inconvenient, so you can instead create a shell that runs in the managed
environment like so:

```bash
pipenv shell
```

and then commands like `pycodestyle` and `pytest` can be run like normal.

Additionally, we use [Travis CI](https://travis-ci.com/ubclaunchpad/mimic) as
a CI system. To run the same checks locally, we provide `scripts/build_check.sh`;
this can be run with:

```bash
./scripts/build_check.sh
```