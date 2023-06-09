# Inflam
Inflam is a data management system written in Python that manages trial data used in clinical inflammation studies.

![Continuous Integration build in GitHub Actions](https://github.com/Berryti1/python-intermediate-inflammation/workflows/CI/badge.svg?branch=main)


## Main features
Here are some key features of Inflam:

- Provide basic statistical analyses over clinical trial data
- Ability to work on trial data in Comma-Separated Value (CSV) format
- Generate plots of trial data
- Analytical functions and views can be easily extended based on its Model-View-Controller architecture

## Prerequisites
Inflam requires the following Python packages:

- [NumPy](https://www.numpy.org/) - makes use of NumPy's statistical functions
- [Matplotlib](https://matplotlib.org/stable/index.html) - uses Matplotlib to generate statistical plots

The following optional packages are required to run Inflam's unit tests:

- [pytest](https://docs.pytest.org/en/stable/) - Inflam's unit tests are written using pytest
- [pytest-cov](https://pypi.org/project/pytest-cov/) - Adds test coverage stats to unit testing

## Basic Usage
Run inflammation-analysis.py with input files as arguments
e.g. python inflammation 

Licensed under GNU General Public License v3.0 2023 by authors
