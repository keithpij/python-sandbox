### About
TODO 

### Creating a Virtual Environment
It is a convention in the Python community to create a virtual environment named '.venv' in the root of your repo.
To create a virual environment named '.venv' use the following command below. 

python -m venv .venv

### Activating your new Virtual Environment
To activate your new virtual environment use one of the following commands based on your operating system.

MacOS or Linux:
    source .venv/bin/activate

Windows:
    source .venv/Scripts/activate

### Deactivating a Virtual Environment
To deactivate:
    deactivate

### Installing packages used by one of the experiments in this repo.
To install all the packages in the requirements.txt file run the folling command once you have 
activated your virtual environment:

    pip install -r requirements.txt

### Download the Large Movie Review Dataset
Download the Large Movie Review Dataset:
    https://ai.stanford.edu/~amaas/data/sentiment/
    
Unzip the downloaded tar.gz file and move the aclImdb folder to the folder containing this code.

### A Tip on running Jupyter notebooks within VS Code
If you are using VS Code to run Jupyter notebooks then you will need to select a kernel for your notebooks. 
(This is the computer icon shown at the top right of each notebook.) You can use a virtual environment as your kernel
only if you locate it at the root of your repo. If you create a virtual environment within a subfolder then VS Code will not be able to find it in the dialog used for selecting kernel. So be sure to create your virtual envirnoment at the root.