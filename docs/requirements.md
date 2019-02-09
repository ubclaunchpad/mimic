# Requirements

## MVP

Our MVP will be a library with a relatively simple interface that can:

* Take a corpus as input, or provide a sample corpus (e.g. Shakespeare,
  "talk like a pirate", Reddit or Twitter data, etc.).
* Generate text from a chosen model(s), such as LSTM RNN, Markov chain, or
  others.

## Stretch Goals

* Building out a whole chatbot system, where the library user can configure
  options, set up stock responses/topics, etc, might be a good direction to go
  with this, though that would lean more into the software engineering than the
  ML side of things.
* More ''smarts'' might be nice - for our MVP it's probably enough to simply
  generate a random line or two of text at a time, but it would be nice to have
  features related to topic modeling, or maybe video description, or...
* For now, we should build this as a Python library - however, we may
  want to expose an HTTP API in the future.

## Non-functional & Other Requirements

* All code will follow the [PEP8 style guide](http://pep8.org);
  this will be automated with
  [pycodestyle](https://github.com/pycqa/pycodestyle).
* There should be automated tests for most user-visible behaviour,
  run with a CI system, and code coverage should be collected and uploaded to
  [Codecov.io](https://codecov.io).
