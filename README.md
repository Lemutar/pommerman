# Pommerman

## What is it?
Attempts to solve the Pommerman challenge

## Setup and configuration

### Installation
The analysis code requires both Python 3.6.X and `pip3` (Python package manager) installed.  On Mac OS X, the simplest way to do this is to install `python3` with [homebrew](https://brew.sh/)
```bash
brew install python3
```
which also installs `pip`. Next, install `virtualenv`
```bash
pip install virtualenv
```
and fire up a python virtual environment as follows
```bash
virtualenv -p python env
```
Finally, activate the environment
```bash
source env/bin/activate
```
and install the required libraries:
```bash
# for running without GPU
pip install -r ./requirements-cpu.txt
# for running with GPU
pip install -r ./requirements-gpu.txt
```

## Setup for gc ray cluster
```bash
export GOOGLE_APPLICATION_CREDENTIALS=<your cred file>
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

ray up ./gc-ray-setup.yaml

ray exec ./gc-ray-setup.yaml 'python ./pommerman/pommber_run_gc.py'

ray down gc-ray-setup.yaml
```
