#!/usr/bin/env bash
set -euxo pipefail
pipenv run pycodestyle .
pipenv run pydocstyle .
mdl .
pipenv run pytest mimic/tests/ --cov=./ --cov-branch --cov-config .coverageac
