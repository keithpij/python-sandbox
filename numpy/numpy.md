Experiments with various numpy utilities.

Numpy has been modified quite a bit over the years and if you are not runnning the latest version then there will be features in the online docmentation that will not work for olderver versions.

To determine the version of numpy you are using use the following code.

import numpy as np

print('Numpy version:  ' + np.version.version)

if it is less than 1.13.1 then you are using an older version of numpy. One of the commands below will upgrade numpy for you. If you are using a Mac and you installed Python 3.X alongside Python 2.7 then you may need to run the pip3 command to upgrade numpy for Python 3.X.

pip install --upgrade numpy
pip3 install --upgrade numpy