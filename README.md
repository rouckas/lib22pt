# lib22pt
Libraries for AB-22PT data analysis

## Basic usage
To analyse a rate measurement in file `r1006_06.dat`, you need to import the `rate` module:
```
from lib22pt.rate import *
```
load the file using class `Rate`:
```n
R = Rate("r1006_06.dat")
```
Assuming that you want to fit a decay curve, use the `fit_decay` method to do the fit:
```
p, sigma, pval = R.fit_decay()
```
The method returns the fit parameters stored in `p`,  error estimates in `sigma`, and the fit p-value (goodness of fit) in `pval`. The fit has two parameters - initial number `p[0]` and decay rate `p[1]`. The corresponding errors are `sigma[0]` and `sigma[1]`. To print the rate you can use the following C-style formatted print command:
```python
print("r = (%.4g +/- %.2g) s^-1" % (p[1], sigma[1]))
```
> r = (21.05 +/- 0.45) s^-1

The fitted rate must be divided by the reactant number density to obtain the reaction rate coefficient. It can be obtained from the measured pressure in the chamber `Paml22PT`, the trap temperature `temp` and the gas-specific calibration factor `f_22PT` using the function `concentration` from the `rate` module. Assuming a reaction with H2 and the following experimental parameters:
```
T22 = 314.          # K
Paml_H2 = 1.4e-8    # Pa
Paml_H2_SIS = 0.    # Pa
f_H2_SIS = 38       # calibration from 2015-03-24
```
the number density is
```
n_H2 = concentration(T22, Paml_H2, Paml_H2_SIS, f_H2_SIS)
print("%.3g cm^-3" % n_H2)
```
> 1.26e+10 cm^-3

and the rate coefficient is:
```
k, k_err = p[1]/n_H2, sigma[1]/n_H2
print("k = (%.4g +/- %.2g) cm^3 s^-1" % (k, k_err))
```
> k = (1.676e-09 +/- 3.6e-11) cm^3 s^-1

But to get correct result, we need to subtract the number density...

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
