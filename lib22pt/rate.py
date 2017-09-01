import numpy as np

# 2013-10-31 Added MultiRate class, simplified fitting methods, removed full_output parameter
# 2014-12-18 Add loading of Frequency, Integration time and Iterations, calculate lower
#            bound on errors from Poisson distribution
# 2015-01-28 Simplified fitting again. Needs more work
# 2015-03-02 Added functions for number density calculation

_verbosity = 2
def set_verbosity(level):
    """
        0: serious/unrecoverable error
        1: recoverable error
        2: warning
        3: information
    """
    global _verbosity
    _verbosity = level

def warn(message, level):
    if level <= _verbosity:
        print(message)


def fitter(p0, errfunc, args):

    from lmfit import minimize
    result = minimize(errfunc, p0, args=args)

    if not result.success:
        msg = " Optimal parameters not found: " + result.message
        raise RuntimeError(msg)

    for i, name in enumerate(result.var_names):
        if result.params[name].value == result.init_vals[i]:
            warn("Warning: fitter: parameter \"%s\" was not changed, it is probably redundant"%name, 2)

    from scipy.stats import chi2
    chi = chi2.cdf(result.redchi, result.nfree)
    if chi > 0.5: pval = -(1-chi)*2
    else: pval = chi*2
    pval = chi
    return result.params, pval, result

def dict2Params(dic):
    from lmfit import Parameters
    if isinstance(dic, Parameters): return dic
    p = Parameters()
    for key, val in dic.items():
        p.add(key, value=val)
    return p
P = dict2Params


class Rate:
    def __init__(self, fname, full_data=False, skip_iter=[]):
        import re
        import datetime as dt
        fr = open(fname)

        state = -1
        npoints = 0
        nions = 0

        pointno = 0
        iterno = 0

        ioniter = []
        ionname = []
        frequency = 0
        integration = 0

        poisson_error = True
        # -1 header
        # 0 init
        # 1 read time
        # 2 read data

        for lineno, line in enumerate(fr):
            # read header
            if state == -1:
                if lineno == 2:
                    T1 = line[:22].split()
                    T2 = line[22:].split()
                    self.starttime = dt.datetime.strptime(" ".join(T1), "%Y-%m-%d %H:%M:%S.%f")
                    self.stoptime = dt.datetime.strptime(" ".join(T2), "%Y-%m-%d %H:%M:%S.%f")
                if lineno == 3:
                    state = 0

            toks = line.split()
            if len(toks) == 0:
                continue
            if state == 0:
                if re.search("Frequency=", line):
                    frequency = float(re.search("Frequency=([0-9.]+)", line).group(1))
                if re.search("Integration time \(s\)", line):
                    integration = float(re.search("Integration time \(s\)=([0-9.]+)", line).group(1))
                if re.search("Number of Points=", line):
                    npoints = int(re.search("Number of Points=(\d+)", line).group(1))
                if re.search("Number of Iterations=", line):
                    self.niter = int(re.search("Number of Iterations=(\d+)", line).group(1))
                if toks[0] == "[Ion":
                    nions += 1
                if re.search("^Iterations=", line) :
                    ioniter.append(int(re.search("Iterations=(\d+)", line).group(1)))
                if re.search("^Name=", line) :
                    ionname.append(re.search("Name=(.+)$", line).group(1))

            if toks[0] == "Time":
                if len(toks)-2 != nions:
                    print("Corrupt file", fname, "Wrong number of ions in the header. Trying to recover")
                    # Assume that the Time header is correct:
                    nions = len(toks)-2
                    ioniter = ioniter[:nions]

                if len(ioniter) < nions:
                    warn("Corrupt file " + str(fname) + ": Iterations for all species not recorded, guessing...", 1)
                    while len(ioniter) < nions:
                        ioniter.append(ioniter[-1])

                if len(ionname) < nions:
                    warn("Corrupt file " + str(fname) +  ": Names for all species not recorded, making something up...", 2)
                    while len(ionname) < nions:
                        ionname.append("Ion%d" % (len(ionname)+1,))

                state = 1
                time = []
                data = np.zeros((nions, npoints, self.niter))
                continue

            if state == 1:
                try:
                    newtime = float(toks[0])
                except ValueError:
                    if pointno != npoints:
                        warn("Corrupt file " + fname + " trying to guess number of points", 2)
                        npoints = pointno
                        data.resize((nions, npoints, self.niter)) 
                    time = np.array(time)
                    state = 2
                else:
                    time.append(newtime)
                    pointno += 1

            if state == 2:
                if toks[0] == "Iteration":
                    iterno = int(toks[1])-1
                    if iterno+1 > self.niter:
                        warn("Corrupt file " + fname + " trying to guess number of iterations", 2)
                        #msg = "Corrupt file: " + fname
                        #raise IOError(msg)
                        self.niter = iterno+1
                        data.resize((nions, npoints, self.niter)) 
                    pointno = 0
                    continue
                data[:, pointno, iterno] = [float(x) for x in toks][1:-1]
                pointno += 1

        ioniter = np.array(ioniter)
        # in case of multiple measurements per iteration
        if iterno+1 != self.niter:
            if self.niter % (iterno+1) != 0:
                msg = "Corrupt file: " + fname
                print(("Corrupt file " + fname + " trying to guess number of iterations:" + str(iterno+1)))
                if iterno+1 < self.niter:
                    data = data[:,:,:iterno+1]
                else:
                    newdata = np.zeros((nions, npoints, iterno+1))
                    newdata[:,:,:self.niter] = data
                    print(data, newdata)
                    data = newdata
                    #data.resize((nions, npoints, iterno+1))
                self.niter = iterno+1
            data = data[:,:,:iterno+1]
        
        #print skip_iter, np.shape(skip_iter)
        if len(skip_iter)!=0:
            skip_iter = np.array(skip_iter)
            indices = np.ones(self.niter, dtype=bool)
            indices[skip_iter] = False
            data = data[:,:,indices]

        # XXX frequency is sometimes wrong in the files
        # use some heuristics to estimate the frequency
        # repetition time is usually set in multiples of 0.1s
        measurement_time = np.ceil(time[-1]/0.1)*0.1
        if frequency*measurement_time > 1.1 or frequency*measurement_time < 0.4:
            warn("Recorded frequency in " + fname + " is probably wrong. Using estimate %f" % (1/measurement_time), 1)
            frequency = 1/measurement_time
        # this is later used to estimate Poisson error
        self.total_iterations = ioniter[:,None]*integration*frequency*self.niter

        self.nions = nions
        self.ionname = ionname

        self.time = time
        self.data = data

        self.average()

        if not full_data:
            self.data = None
        self.mask = None


    def average(self):
        data_mean = np.mean(self.data, axis=2)
        data_std = np.std(self.data, axis=2)/np.sqrt(self.niter)

        #print(np.shape(self.data), np.shape(data_mean), np.shape(self.total_iterations))
        data_counts = data_mean*self.total_iterations
        # divide by sqrt(total_iterations) twice - once to get Poisson
        # variance of measured data and once to the error of estimated mean
        # this should be verified, but it is in agreement with errors obtained
        # by treating data as normal variables for large numbers
        data_poiss_err = np.sqrt(np.maximum(data_counts, 3))/self.total_iterations
        # we assume that if 0 counts are observed, 3 counts is within confidence interval

        # we use std error if it is larger than poisson error to capture other sources
        # of error e.g. fluctuations
        data_std = np.maximum(data_std, data_poiss_err)

        self.data_mean = data_mean
        self.data_std = data_std

    def merge(self, rate2):
        self.data_mean = np.concatenate((self.data_mean, rate2.data_mean), axis=1)
        self.data_std = np.concatenate((self.data_std, rate2.data_std), axis=1)
        self.time = np.concatenate((self.time, rate2.time), axis=0)
        #print " ** merging ** "
        #print self.data_mean, self.data_std, self.time



    def poisson_test1(self):
        shape = np.shape(self.data_mean)
        #check only H- XXX
        shape = (1, shape[1])
        pval = np.zeros(shape)
        for specno in range(shape[0]):
            for pointno in range(shape[1]):
                if self.mask != None:
                    dataline = self.data[specno, pointno, self.mask[specno, pointno, :]]
                else:
                    dataline = self.data[specno, pointno, :]
                mean = np.mean(dataline)
                Q = np.sum((dataline-mean)**2)/mean
                niter = len(dataline[~np.isnan(dataline)])
                dof = niter-1
                from scipy.stats import chi2
                chi = chi2.cdf(Q, dof)
                if chi > 0.5: pval[specno, pointno] = (1-chi)*2
                else: pval[specno, pointno] = chi*2
                print((chi, Q, pval[specno, pointno]))
        return np.min(pval)


    def cut3sigma(self, nsigma=3):
        shape = np.shape(self.data)
        self.mask = np.zeros(shape, dtype=bool)
        for specno in range(shape[0]):
            for pointno in range(shape[1]):
                stddev = self.data_std[specno, pointno]*np.sqrt(self.niter)
                low = self.data_mean[specno, pointno] - nsigma*stddev
                high = self.data_mean[specno, pointno] + nsigma*stddev
                dataline = self.data[specno, pointno, :]
                mask = (dataline > low) & (dataline < high)
                #self.data[specno, pointno, ~mask] = float("nan")
                self.mask[specno, pointno, :] = mask
                self.data_mean[specno, pointno] = np.mean(dataline[mask])
                self.data_std[specno, pointno] = np.std(dataline[mask])/np.sqrt(self.niter)
        #data_mean = np.mean(self.data[self.mask], axis=2)
        #data_std = np.std(self.data, axis=2)/np.sqrt(self.niter)
        #print self.data_mean, self.data_std

        #self.data[self.data<120] = 130


    def fit_ode_mpmath(self, p0=[60.0, .1], columns=[0]):
        from mpmath import odefun
        def fitfunc(p, x):
            eqn = lambda x, y: -p[1]*y
            y0 = p[0]
            f = odefun(eqn, 0, y0)
            g = np.vectorize(lambda x: float(f(x)))
            return g(x)

        return self._fit(fitfunc, p0, columns)


    def fit_ode_scipy(self, p0=[60.0, .1], columns=[0]):
        from scipy.integrate import odeint
        def fitfunc(p, x):
            eqn = lambda y, x: -p[1]*y
            y0 = p[0]
            t = np.r_[0., x]
            y = odeint(eqn, y0, t)
            return y[1:,0]
        return self._fit(fitfunc, p0, columns)


    def fit_inc(self, p0=[1.0, .01, 0.99], columns=[1]):
        #fitfuncinc = lambda p, x: p[0]*(1-np.exp(-x/p[1]))+p[2]
        fitfunc = lambda p, x: -abs(p[0])*np.exp(-x/abs(p[1]))+abs(p[2])
        return self._fit(fitfunc, p0, columns)
    
    def fit_equilib(self, p0=[70.0, .1, 1], columns=[0]):
        fitfunc = lambda p, x: abs(p[0])*np.exp(-x/abs(p[1]))+abs(p[2])
        return self._fit(fitfunc, p0, columns)



class MultiRate:
    def __init__(self, fnames, directory=""):
        if isinstance(fnames, str): fnames = [fnames]
        self.rates = [Rate(directory+fname, full_data=True) for fname in fnames]

        # if True, a normalization factor for each rate with respect to rates[0] is a free fitting param
        self.normalized = True
        self.norms = [1]*len(self.rates)

        self.fitfunc = None
        self.fitparam = None
        self.fitcolumns = None
        self.fitmask = slice(None)

    def plot(self, ax=None, show=False, plot_fitfunc=True, symbols=["o", "s", "v", "^", "D", "h"], colors=["r", "g", "b", "m", "k", "orange"],\
            opensymbols=False, fitfmt="-", fitcolor=None, hide_uncertain=False, plot_columns=None):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()

        lines = {}
        if plot_columns is None: plot_columns = range(self.rates[0].nions)
        for i in plot_columns:
            if opensymbols:
                kwargs = {"markeredgewidth":1, "markerfacecolor":"w", "markeredgecolor": colors[i], "color":colors[i]}
            else:
                kwargs = {"markeredgewidth":0, "color":colors[i]}
            l = None
            for j, rate in enumerate(self.rates):
                norm = 1/self.norms[j]

                I = rate.data_std[i] < rate.data_mean[i] if hide_uncertain else slice(None)
                if l==None:
                    l = ax.errorbar(rate.time[I], rate.data_mean[i][I]*norm, yerr=rate.data_std[i][I]*norm, label=rate.ionname[i],
                        fmt = symbols[i], **kwargs)
                    color = l.get_children()[0].get_color()
                else:
                    l = ax.errorbar(rate.time[I], rate.data_mean[i][I]*norm, yerr=rate.data_std[i][I]*norm,
                        fmt = symbols[i], color=color, markeredgewidth=0)
            lines[i] = l

        # plot sum
        for j, rate in enumerate(self.rates):
            # calculate the sum over the plotted data only
            S = np.sum(rate.data_mean[plot_columns], axis=0)
            label = "sum" if j==0 else None
            ax.plot(rate.time, S/self.norms[j], ".", c="0.5", label=label)


        if self.fitfunc != None:
            mintime = np.min([np.min(r.time[self.fitmask]) for r in self.rates])
            maxtime = np.max([np.max(r.time[self.fitmask]) for r in self.rates])
            x = np.logspace(np.log10(mintime), np.log10(maxtime), 500)-self.fit_t0
            x = x[x>=0.]

            fit = self.fitfunc(self.fitparam, x)
            for i, column in enumerate(self.fitcolumns):
                if column not in plot_columns: continue
                if fitcolor == None: c = lines[column].get_children()[0].get_color()
                else: c = fitcolor
                ax.plot(x+self.fit_t0, fit[i], fitfmt, c=c)
            if len(self.fitcolumns) > 1:
                ax.plot(x+self.fit_t0, np.sum(fit, axis=0), c="k")

        if show == True:
            ax.set_yscale("log")
            ax.legend()
            plt.show()

        return ax


    def save_data(self, filename):
        to_save = []
        for j, rate in enumerate(self.rates):
            norm = 1/self.norms[j]
            to_save.append(np.hstack((rate.time[:,np.newaxis], rate.data_mean.T, rate.data_std.T)))
        to_save = np.vstack(to_save)
        np.savetxt(filename, to_save)


    def save_fit(self, filename, time = None):
        if time is None:
            mintime = np.min([np.min(r.time[self.fitmask]) for r in self.rates])
            maxtime = np.max([np.max(r.time[self.fitmask]) for r in self.rates])
            time = np.logspace(np.log10(mintime), np.log10(maxtime), 500)-self.fit_t0
            time = time[time>=0.]

        fit = self.fitfunc(self.fitparam, time)
        to_save = np.vstack((time, np.vstack(fit))).T
        np.savetxt(filename, to_save)


    def save_data_fit_excel(self, filename, time = None):
        import pandas as pd
        for j, rate in enumerate(self.rates):
            if j>0:
                TODO()
                norm = 1/self.norms[j]

            df = pd.DataFrame(rate.time, columns=["tt"])
            for k, name in enumerate(rate.ionname):
                df[name] = rate.data_mean[k]
                df[name+"_err"] = rate.data_std[k]

        writer = pd.ExcelWriter(filename)
        df.to_excel(writer, "data")

        if self.fitfunc != None:
            if time is None:
                mintime = np.min([np.min(r.time[self.fitmask]) for r in self.rates])
                maxtime = np.max([np.max(r.time[self.fitmask]) for r in self.rates])
                time = np.logspace(np.log10(mintime), np.log10(maxtime), 500)-self.fit_t0
                time = time[time>=0.]

            fit = self.fitfunc(self.fitparam, time)

            df_fit = pd.DataFrame(time, columns = ["tt"])
            for i, column in enumerate(self.fitcolumns):
                name = self.rates[0].ionname[column]
                df_fit[name] = fit[i]

            df_fit.to_excel(writer, "fit")
        writer.save()


    def _errfunc(self, p, bounds={}):

        def errfunc_single( p, x, y, xerr, norm=1):
            sigma_min = 0.01
            res = (self.fitfunc(p, x)*norm-y[self.fitcolumns,:])/\
                (xerr[self.fitcolumns,:] + sigma_min)
            return res.ravel()

        # sum errors over all files (normalize if requested)
        err = []
        mask = self.fitmask
        for i, rate in enumerate(self.rates):
            norm = p["n%d"%i].value if (i>0 and self.normalized) else 1
            err.append(errfunc_single(p, rate.time[mask] - self.fit_t0, rate.data_mean[:,mask], rate.data_std[:,mask], norm))

        # add penalty for bounded fitting
        penalty = [weight*(np.fmax(lo-p[key], 0) + np.fmax(0, p[key]-hi))\
                    for key, (lo, hi, weight) in bounds.items()]
        err.append(penalty)

        #print("err = ", np.sum(np.hstack(err)**2))
        return np.hstack(err)


    def _errfunc_species(self, p, species, bounds={}):

        def errfunc_single( p, x, y, xerr, norm=1):
            sigma_min = 0.01
            res = (self.fitfunc(p, x)[species]*norm-y[self.fitcolumns,:][species])/\
                (xerr[self.fitcolumns,:][species] + sigma_min)
            return res.ravel()

        # sum errors over all files (normalize if requested)
        err = []
        mask = self.fitmask
        for i, rate in enumerate(self.rates):
            norm = p["n%d"%i].value if (i>0 and self.normalized) else 1
            err.append(errfunc_single(p, rate.time[mask] - self.fit_t0, rate.data_mean[:,mask], rate.data_std[:,mask], norm))

        # add penalty for bounded fitting
        penalty = [weight*(np.fmax(lo-p[key], 0) + np.fmax(0, p[key]-hi))\
                    for key, (lo, hi, weight) in bounds.items()]
        err.append(penalty)

        #print("err = ", np.sum(np.hstack(err)**2))
        return np.hstack(err)


    def fit_model(self, model, columns, mask=slice(None), t0=0., boundweight=1e3):
        self.fitfunc = model.func # store the fitfunc for later
        self.model = model
        self.fitcolumns = columns
        self.fit_t0 = t0
        self.fitmask = mask

        if type(model.params) is dict:
            p0 = dict2Params(model.params)
        else:
            p0 = model.params.copy()


        # use custom implementation of bounded parameters, which can
        # estimate errors even if the parameter is "forced" outside the interval
        # idea: detect if fitted parameter is close to boundary, then make it fixed...
        bounds = {}
        for key, p in p0.items():
            if np.any(np.isfinite([p.min, p.max])):
                bounds[key] = (p.min, p.max, boundweight)
                p.set(min=-np.inf, max=np.inf)

        # If needed, the normalization factors are appended to the list of
        # fitting parameters
        if self.normalized:
            for i in range(1,len(self.rates)): p0.add("n%d"%i, value=1)

        # DO THE FITTING
        self.fitparam, pval, result = fitter(p0, self._errfunc, (bounds,))

        # extract the normalization factors
        if self.normalized:
            for i in range(1,len(self.rates)): self.norms[i] = self.fitparam["n%d"%i].value

        return self.fitparam, pval, result


    def fit_decay(self, p0={"N0":100, "r":10}, columns=[0], mask=slice(None), t0=0.):
        from .fitmodels import Decay
        return self.fit_model(Decay().set_params(p0), columns, mask, t0)


    def fit_change(self, p0={"N0":100, "N1": 10, "r":10}, columns=[0,1], mask=slice(None), t0=0., loss=0.):
        from .fitmodels import Change
        p0 = dict2Params(p0)
        p0.add("loss", value=loss, vary=False)
        return self.fit_model(Change().set_params(p0), columns, mask, t0)


    def fit_change_channel(self, p0 = P({"N0": 1000, "N1": 100, "r": 10, "bratio":.5}),\
            columns=[0,1], mask=slice(None), t0=0.):
        from .fitmodels import ChangeChannel
        return self.fit_model(ChangeChannel().set_params(p0), columns, mask, t0)


    def fit_change_channel_bg(self, p0 = P({"N0": 1000, "N1": 100, "r": 10, "bratio":.5, "bg": 1.}),\
            columns=[0,1], mask=slice(None), t0=0.):
        from .fitmodels import ChangeChannelBG
        return self.fit_model(ChangeChannelBG().set_params(p0), columns, mask, t0)


    def fit_change_2channel(self, p0 = P({
        "N0" : 1000.,      "N1": 100.,
        "N2" : 1.,         "r1": 1.,
        "r2" : 10.}),\
            columns=[0,1,2], mask=slice(None), t0=0):
        from .fitmodels import Change2Channel
        return self.fit_model(Change2Channel().set_params(p0), columns, mask, t0)


    def fit_equilib(self, p0 = P({
        "N0" : 1000.,      "N1": 100.,
        "r0" : 1.,         "r1": 1.}),
            columns=[0,1,2], mask=slice(None), t0=0):
        from .fitmodels import Equilib
        return self.fit_model(Equilib().set_params(p0), columns, mask, t0)


    def fit_NH(self, p0=P({
        "N": 100.,          "NH": 10.,
        "NH2": 1.,          "NH3": 1.,
        "H3": 10.,          "N15":10.,
        "rNH":1.,          "rNH2":10.,
        "rH3":1,            "rNH3":10,
        "rH3d":1}),\
            columns=[0,1,2,3,4], mask=slice(None), t0=0, H3disc=1., NH2disc=1., NH3disc=1.):
        from .fitmodels import NH
        p0.add("H3disc", value=H3disc, vary=False)
        p0.add("NH2disc", value=NH2disc, vary=False)
        p0.add("NH3disc", value=NH3disc, vary=False)
        for key in p0: p0[key].set(min=0)
        return self.fit_model(NH().set_params(p0), columns, mask, t0)



    def fit_NHn_long(self, p0=P({
        "N": 100.,          "NH": 10.,
        "NH2": 1.,          "NH3": 1.,
        "NH4": .1,          "H3": 10.,
        "rNH":1.,           "rNH2":10.,
        "rH3":1,            "rNH3":10,
        "rNH4":1,           "rH3d":1,
        "rNH3rel":1,        "rNH4exc":10}),\
            columns=[0,1,2,3,4,5], mask=slice(None), t0=0, NH3loss=0.05, H3disc=1.):
        from .fitmodels import NHn_long
        p0.add("H3disc", value=H3disc, vary=False)
        p0.add("NH3loss", value=NH3loss, vary=False)
        for key in p0: p0[key].set(min=0)
        return self.fit_model(NHn_long().set_params(p0), columns, mask, t0)



    def fit_NHn_short(self, p0=P({
        "N":400.,       "NH":.1,
        "NH2":.1,       "NH3":.1,
        "H3":.1,        "rNH":10.,
        "rNH2":100.,    "rNH3":100.,
        "rNH4":10.,     "rH3":10,
        "rH3d":10.
        }),\
            columns=[0,1,2,3,5], mask=slice(None), t0=0, H3disc = 1.):
        from .fitmodels import NHn_short
        p0.add("H3disc", value=H3disc, vary=False)
        for key in p0: p0[key].set(min=0)
        return self.fit_model(NHn_short().set_params(p0), columns, mask, t0)

