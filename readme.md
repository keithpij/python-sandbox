### Useful pyenv commands:

brew install pyenv
brew install pyenv-virtualenv

pyenv install --list
pyenv versions
pyenv global 3.8.5
pyenv local 3.8.5
pyenv global system
pyenv local system
pyenv which pip

### To use an environment in the current shell.
pyenv shell 3.10.0

### If the environment does not seem to be working then try the command below.
eval "$(pyenv init --path)"

### To setup a pyenv virtual environment
pyenv virtualenv 3.8.5 onnx
cd onnx
pyenv local onnx

### To setup a python virtual environment based on a pyenv environment:
pyenv shell 3.10.0
python -m venv .venv
source .venv/bin/activate