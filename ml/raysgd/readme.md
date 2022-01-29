It is a convention in the Python community to create a virtual environment named '.venv' in the root of your repo.
To create a virual environment named '.venv' use the following command below. 

python -m venv .venv

To activate your new virtual environment on a Mac or Linux based machine use the following command.

    source .venv/bin/activate

On Windows use:


To deactivate:
    deactivate

Download the Large Movie Review Dataset:
    https://ai.stanford.edu/~amaas/data/sentiment/


Unzip the downloaded tar.gz file and move the aclImdb folder to the folder containing this code.

For Python environments:
pyenv shell ray