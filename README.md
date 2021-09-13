# DSPS_GFoote
Work for the PHYS 661 class at the University of Delaware for the Fall 2021 Semester

## Google Collab
These projects will ultimately be connected to a set of Google Collab Documents 
which will represent the final stage of the code. There will be a table created here which will list those
at the end of development

## Dependency Management
To manage dependencies, there will be a general [pipenv](https://pipenv.pypa.io/en/latest/) 
environment with the appropriate setup scripts to build the ijuypter kernel for all notebooks. 

### Note
If there is a need for dependency management inside the Google Collab, it will be handled internally by the code which 
will set up an environment and install the appropriate packages with a single run of a cell using information from
[this notebook](https://colab.research.google.com/github/aviadr1/learn-advanced-python/blob/master/content/10_package_management/10-package_management.ipynb#scrollTo=H7DX84ep1dm6)

## Interaction with @fedhere repo
The code in this repository will be directly connected to the professor of this class's repo found [here](https://github.com/fedhere/DSPS_FBianco). 
Forking this repository would mean not being able to use the dedicated name for this repo, which would mess up the [gitallrepos](https://github.com/fedhere/DSPS_FBianco/blob/master/gitallrepos.py)
and [pullallgits](https://github.com/fedhere/DSPS_FBianco/blob/master/pullallgits.py) scripts used to automate the submission of the work.
Because the scripts use `git pull`, treating this parent directory as a submodule is the best course of action. 

## Running Instructions
Open the terminal in the Top Level Directory of this repo and run the following commands in order.

* `pipenv install`
* `pipenv shell`
* `pipenv run jupyter notebook`

## Final Submission Table

Under Construction
