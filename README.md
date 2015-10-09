# lib22pt
Libraries for AB-22PT data analysis

## Basic usage
To analyse a rate measurement in file `r0101_01.dat`, you need to import the `rate` module:
```
from lib22pt.rate import *
```
load the file using class `Rate`:
```n
R = Rate("r0101_01.dat")
```
Assuming that you want to fit a decay curve, use the `fit_decay` method to do the fit:
```
p, sigma, pval = R.fit_decay()
```
The method returns the fit parameters stored in `p`,  error estimates in `sigma`, and the fit p-value (goodness of fit) in `pval`. The fit has two parameters - initial number `p[0]` and decay rate `p[1]`. The corresponding errors are `sigma[0]` and `sigma[1]`.

## Installation
To install for development:

    python3 setup.py develop

Regular installation:

    python3 setup.py install

Local installation:

    copy the lib22pt folder into working directory. Recommended way to make
    "self-contained" data analysis folders.

To uninstall:

    pip3 uninstall lib22pt
