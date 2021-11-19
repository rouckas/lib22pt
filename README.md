# lib22pt
Libraries for AB-22PT data analysis

## Basic usage
To analyse a rate measurement in file `r1006_06.dat`, you need to import the `rate` module:
```python
>>> from lib22pt.rate import *
```
load the file using class `Rate`:
```python
>>> R = MultiRate("r1006_06.dat")
```
Assuming that you want to fit a decay curve, use the `fit_decay` method to do the fit:
```python
>>> param, pval, fitresult = R.fit_decay()
```
The method returns the fit parameters stored in `param`, the fit p-value (goodness of fit) in `pval`, and the `lmfit MinimizerResult` object in fitresult. The fit has two parameters - initial number `N0` and decay rate `r`. The fitted parameter values and estimated errors can be printed as
```python
>>> param.pretty_print()
Name     Value      Min      Max   Stderr     Vary     Expr Brute_Step
N0     142.2     -inf      inf    2.326     True     None     None
r      4.522     -inf      inf   0.2545     True     None     None
```
and the values can be extracted as
```python
>>> print("r = (%.4g +/- %.2g) s^-1" % (param["r"].value, param["r"].stderr))
r = (21.05 +/- 0.45) s^-1
```

The fitted rate must be divided by the reactant number density to obtain the reaction rate coefficient. It can be obtained from the measured pressure in the chamber `Paml22PT`, the trap temperature `temp` and the gas-specific calibration factor `f_22PT` using the function `concentration` from the `rate` module. Assuming a reaction with H2 and the following experimental parameters:
```python
T22 = 314.          # K
Paml_H2 = 1.4e-8    # Pa
Paml_H2_SIS = 0.    # Pa
f_H2_SIS = 38       # calibration from 2015-03-24
```
the number density is
```python
>>> from lib22pt.util import concentration
>>> n_H2 = concentration(T22, Paml_H2, Paml_H2_SIS, f_H2_SIS)
>>> print("%.3g cm^-3" % n_H2)
1.26e+10 cm^-3
```

and the rate coefficient is:
```python
>>> k, k_err = param["r"].value/n_H2, param["r"].stderr/n_H2
>>> print("k = (%.4g +/- %.2g) cm^3 s^-1" % (k, k_err))
k = (1.676e-09 +/- 3.6e-11) cm^3 s^-1
```

## Installation
The recommended way to make "self-contained" data analysis folders is to copy the lib22pt folder into working directory.

Regular installation:

    python3 setup.py install

To install for development:

    python3 setup.py develop

To uninstall:

    pip3 uninstall lib22pt

Run tests:

    python3 setup.py test

