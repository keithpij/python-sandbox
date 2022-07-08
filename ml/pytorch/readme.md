
### Set up a virtual environment

It is a convention in the Python community to create a virtual environment named '.venv' in the root of your repo.
To create a virual environment named '.venv' use the following command below. 

python -m venv .venv

To activate your new virtual environment on a Mac or Linux based machine use the following command.

    source .venv/bin/activate

On Windows use:
    source .venv/Scripts/activate

To deactivate:
    deactivate

To install all the packages in the requirements.txt file run the folling command once you have activated your virtual environment:

    pip install -r requirements.txt

Download the Large Movie Review Dataset:
    https://ai.stanford.edu/~amaas/data/sentiment/


Unzip the downloaded tar.gz file and move the aclImdb folder to the folder containing this code.

### Using pyenv
to start your shell:
    pyenv shell 3.10.0

If your shell does not seem to be going into effect then try this command:
    eval "$(pyenv init --path)"
