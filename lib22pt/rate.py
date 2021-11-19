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
    result = minimize(errfunc, p0, args=args, nan_policy="omit")

    if not result.success:
        msg = " Optimal parameters not found: " + result.message
        raise RuntimeError(msg)

    for i, name in enumerate(result.var_names):
        if result.params[name].value == result.init_vals[i]:
            warn("Warning: fitter: parameter \"%s\" was not changed, it is probably redundant"%name, 2)

    from scipy.stats import chi2
    chi = chi2.cdf(result.chisqr, result.nfree)
    if chi > 0.5: pval = -(1-chi)*2
    else: pval = chi*2
    pval = 1-chi
    return result.params, pval, result

def dict2Params(dic):
    from lmfit import Parameters
    if isinstance(dic, Parameters): return dic.copy()
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
                if re.search("Period \(s\)=", line):
                    frequency = 1/float(re.search("Period \(s\)=([0-9.]+)", line).group(1))
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
                    ionname.append(re.search("Name=(.+)$", line).group(1).strip('\"'))

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
                    ionname += toks[len(ionname)+2:]

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
                try:
                    data[:, pointno, iterno] = [float(x) for x in toks][1:-1]
                except ValueError:
                    warn("Error in file " + fname + " number of ions probably wrong")
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

        self.fname = fname

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
        self.fitresult = None
        self.fitcolumns = None
        self.fitmask = slice(None)
        self.fnames = fnames

        self.sigma_min = 0.01 # lower bound on the measurement accuracy

    def plot_to_file(self, fname, comment=None, figsize=(6,8.5), logx=False, *args, **kwargs):
        import matplotlib.pyplot as plt
        from lmfit import fit_report

        f = plt.figure(figsize=figsize)
        ax = f.add_axes([.15, .5, .8, .45])
        self.plot(ax=ax, show=False, *args, **kwargs)
        ax.set_yscale("log")
        if logx: ax.set_xscale("log")
        ax.legend(loc="lower right", fontsize=5)
        ax.set_title(comment, size=8)

        if self.fitresult is not None:
            f.text(0.1, 0.44, "p-value = %.2g\n"%self.fitpval
                    + fit_report(self.fitresult, min_correl=0.5), size=6, va="top", family='monospace')
        if ax.get_ylim()[0] < 1e-4: ax.set_ylim(bottom=1e-4)
        ax.set_xlabel(r"$t (\rm s)$")
        ax.set_ylabel(r"$N_{\rm i}$")

        if self.fitresult is not None:
            ax2 = f.add_axes([.55, .345, .40, .10])
            if logx: ax2.set_xscale("log")

            self.plot_residuals(ax=ax2, show=False, weighted=True)
            ax2.tick_params(labelsize=7)
            ax2.set_title("weighted residuals", size=7)
            ax2.set_xlabel(r"$t (\rm s)$", size=7)
            ax2.set_ylabel(r"$R/\sigma$", size=7)


        f.savefig(fname, dpi=200)
        plt.close(f)

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


        if self.fitfunc != None and self.fitparam != None:
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

    def plot_residuals(self, ax=None, show=False, weighted=False, symbols=["o", "s", "v", "^", "D", "h"], colors=["r", "g", "b", "m", "k", "orange"],\
            opensymbols=False, plot_columns=None):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()

        if plot_columns is None: plot_columns = range(self.rates[0].nions)
        cdict = {col: i for i, col in enumerate(plot_columns)}
        lines = {}
        for j, rate in enumerate(self.rates):
            t = rate.time[self.fitmask]
            #print("\n"*3 + "*"*80)
            #print(rate.fname)
            fit = self.fitfunc(self.fitparam, t-self.fit_t0)
            for i, column in enumerate(self.fitcolumns):
                """
                print("\n"*2 + "*"*3 + " " + rate.ionname[column])
                print(rate.time)
                print(t - self.fit_t0)
                print(rate.data_mean[column])
                print(rate.data_std[column])
                print(fit[i])
                print((rate.data_mean[column][self.fitmask] - fit[i])/rate.data_std[column][self.fitmask])
                """
                if column in plot_columns:
                    j = cdict[column]
                    if weighted:
                        ax.plot(t, (rate.data_mean[column][self.fitmask] - fit[i])/rate.data_std[column][self.fitmask],
                                symbols[j], color=colors[j], lw=0.5, ms=2)
                    else:
                        ax.errorbar(t, rate.data_mean[column][self.fitmask] - fit[i], yerr=rate.data_std[column][self.fitmask],
                                fmt=symbols[j], color=colors[j], lw=0.5, ms=2)
        #ax.set_yscale("symlog", linthresh=10)

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
            time = np.logspace(np.log10(mintime), np.log10(maxtime), 500)

        time = time[time-self.fit_t0 >= 0.]
        fit = self.fitfunc(self.fitparam, time-self.fit_t0)
        to_save = np.vstack((time, np.vstack(fit))).T
        np.savetxt(filename, to_save)


    def save_data_fit_excel(self, filename, time=None, normalize=False, metadata={}):
        import pandas as pd
        dfs = []
        for j, rate in enumerate(self.rates):
            df = pd.DataFrame(rate.time, columns=["tt"])
            if self.normalized:
                df["norm"] = 1/self.norms[j]
                if normalize:
                    norm = 1/self.norms[j]
                else:
                    norm = 1

            for k, name in enumerate(rate.ionname):
                df[name] = rate.data_mean[k]*norm
                df[name+"_err"] = rate.data_std[k]*norm
            df["rate"] = rate.fname
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        writer = pd.ExcelWriter(filename)
        df.to_excel(writer, "data")

        if self.fitfunc != None and self.fitparam != None:
            if time is None:
                mintime = np.min([np.min(r.time[self.fitmask]) for r in self.rates])
                maxtime = np.max([np.max(r.time[self.fitmask]) for r in self.rates])
                time = np.logspace(np.log10(mintime), np.log10(maxtime), 500)

            time = time[time-self.fit_t0 >= 0.]
            fit = self.fitfunc(self.fitparam, time-self.fit_t0)

            df_fit = pd.DataFrame(time, columns = ["tt"])
            for i, column in enumerate(self.fitcolumns):
                name = self.rates[0].ionname[column]
                df_fit[name] = fit[i]

            df_fit.to_excel(writer, "fit")

        for key in metadata.keys():
            metadata[key].to_excel(writer, key)

        writer.save()

    def _errfunc(self, p, bounds={}):

        def errfunc_single( p, x, y, xerr, norm=1):
            res = (self.fitfunc(p, x)*norm-y[self.fitcolumns,:])/\
                (xerr[self.fitcolumns,:] + self.sigma_min)
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
            res = (self.fitfunc(p, x)[species]*norm-y[self.fitcolumns,:][species])/\
                (xerr[self.fitcolumns,:][species] + self.sigma_min)
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


    def fit_model(self, model, p0, columns, mask=slice(None), t0=0., quadratic_bounds=True, boundweight=1e3):
        self.fitfunc = model.func # store the fitfunc for later
        self.model = model
        self.fitcolumns = columns
        self.fit_t0 = t0
        self.fitmask = mask

        p0 = model.init_params(p0)

        # use custom implementation of bounded parameters, which can
        # estimate errors even if the parameter is "forced" outside the interval
        # idea: detect if fitted parameter is close to boundary, then make it fixed...
        bounds = {}
        if quadratic_bounds:
            for key, p in p0.items():
                if np.any(np.isfinite([p.min, p.max])):
                    bounds[key] = (p.min, p.max, boundweight)
                    p.set(min=-np.inf, max=np.inf)

        # If needed, the normalization factors are appended to the list of
        # fitting parameters
        if self.normalized:
            for i in range(1,len(self.rates)): p0.add("n%d"%i, value=1)

        # DO THE FITTING
        self.fitparam, self.fitpval, self.fitresult = fitter(p0, self._errfunc, (bounds,))

        # extract the normalization factors
        if self.normalized:
            for i in range(1,len(self.rates)): self.norms[i] = self.fitparam["n%d"%i].value

        return self.fitparam, self.fitpval, self.fitresult


    def fit_decay(self, p0={"N0":100, "r":10}, columns=[0], mask=slice(None), t0=0.):
        from .fitmodels import Decay
        return self.fit_model(Decay(), p0, columns, mask, t0)
