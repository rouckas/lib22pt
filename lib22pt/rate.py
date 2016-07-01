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
    p = Parameters()
    for key, val in dic.items():
        p.add(key, value=val)
    return p


class Rate:
    def __init__(self, fname, full_data=False, skip_iter=[]):
        import re
        fr = open(fname)

        state = 0
        npoints = 0
        nions = 0

        pointno = 0
        iterno = 0

        ioniter = []
        ionname = []
        frequency = 0
        integration = 0

        poisson_error = True
        # 0 init
        # 1 read time
        # 2 read data

        for line in fr:
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

        self.fitfunc = None
        self.fitparam = None
        self.fitcolumns = None
        self.fitmask = slice(None)
        self.fit_t0 = 0

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

    def plot(self, ax=None, show=False, plot_fitfunc=True, symbols=["o", "s", "v", "^", "D", "h"], fitfmt="-", fitcolor=None):
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        lines = []
        for i in range(self.nions):
            l = ax.errorbar(self.time, self.data_mean[i], yerr=self.data_std[i], label=self.ionname[i],
                    fmt = symbols[i])
            lines.append(l)

        if self.fitfunc != None:
            x = np.linspace(np.min(self.time), np.max(self.time))
            if len(self.fitcolumns) > 1:
                for i, column in enumerate(self.fitcolumns):
                    if fitcolor == None: c = lines[column].get_children()[0].get_color()
                    else: c = fitcolor
                    ax.plot(x, self.fitfunc(self.fitparam, x)[i], fitfmt, c=c)
            else:
                column = self.fitcolumns[0]
                if fitcolor == None: c = lines[column].get_children()[0].get_color()
                else: c = fitcolor
                ax.plot(x, self.fitfunc(self.fitparam, x), fitfmt, c=c)

        if show == True:
            ax.set_yscale("log")
            ax.set_title("p="+str(self.fitparam))
            ax.legend()
            plt.show()

    def _fit(self, fitfunc, p0, columns, mask=slice(None), bounds=None):
        self.fitfunc = fitfunc # store the fitfunc for later
        self.fitcolumns = columns
        def errfunc( p, x, y, xerr):
            sigma_min = 0.01
            retval = (fitfunc(p, x)-y[columns,:])/\
                (xerr[columns,:] + sigma_min)
            #print(p, np.sum(retval**2))
            return retval.ravel()
        args=(self.time[mask], self.data_mean[:,mask], self.data_std[:,mask])
        self.fitparam, sigma, pval = self.fitter(p0, errfunc, args, bounds)
        return self.fitparam, sigma, pval


    def fit_decay(self, p0=[60.0, .1], columns=[0], mask=slice(None)):
        fitfunc = lambda p, x: p[0]*np.exp(-x*p[1])
        return self._fit(fitfunc, p0, columns, mask)

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

    def fitspecial(self, p0, columns=[0,1]):
        def fitfuncspec(self, p, t):
            #KfKb = p[0]/p[1]*p[3]
            #y0 = (p[0] - p[1])*np.exp(-KfKb *(t-p[4])) + p[1]
            #y1 = p[0] - ((p[0] - p[1])*np.exp(-KfKb *(t-p[4])) + p[1])
            #y0 = (p[0] - p[1])*np.exp(-(p[2]+p[3])*(t-p[4])) + p[1]
            #y1 = p[0] - ((p[0] - p[1])*np.exp(-(p[2]+p[3])*(t-p[4])) + p[1])
            #p = A0, B0, kf, kb, t0
            t0 = p[4]
            kt = p[2]+p[3]
            y0 = p[0]/kt*(p[3]+p[2]*np.exp(-kt*(t-t0)))  + p[1]*p[3]/kt*(1-np.exp(-kt*(t-t0)))
            y1 = p[0]/kt*p[2]*(1-np.exp(-kt*(t-t0)))     + p[1]/kt*(p[2]+p[3]*np.exp(-kt*(t-t0)))
            return([y0,y1])
            
        #def errfuncspec(self, p, t, y0, yerr0, y1, yerr1):
        #    y_cal = self.fitfuncspec(p, t)
        #    a = (y_cal[0]-y0)/(yerr0+0.001)
        #    b = (y_cal[1]-y1)/(yerr1+0.001)
        #    #return(np.sqrt(a**2+b**2))
        #    #return(np.sqrt(4*a**2+b**2))
        #    return(a+b)

        return self._fit(fitfunc, p0, columns)
        #args= (self.time, self.data_mean[0,:], self.data_std[0,:], self.data_mean[1,:], self.data_std[1,:])
        #return self.fitter(p0, errfuncspec, args)

    def fitOH(self, p0=[260, 10., 1, 0.01], OH_loss=False, OH_injection=False):
    #def fitOH(self, p0=[260, 1., 1., 0.01], full_output=False):
        #fitfuncinc = lambda p, x: p[0]*(1-np.exp(-x/p[1]))+p[2]
        if OH_loss:
            if OH_injection:
                if len(p0)==4:
                    p0 = p0 + [0]
                fitfunc = lambda p, x: (
                        p[0]*np.exp(-x*p[1]), 
                        p[2]*p[0]/(p[1]-p[3])*(np.exp(-x*(p[3]))-np.exp(-x*p[1])) +\
                        (p[4])*np.exp(-x*(p[3]))
                        )
            else:
                fitfunc = lambda p, x: (
                        p[0]*np.exp(-x*p[1]), 
                        p[2]*p[0]/(p[1]-(p[3]))*(np.exp(-x*(p[3]))-np.exp(-x*p[1])) +\
                        0.*np.exp(-x*(p[3]))
                        )
        else:
            fitfunc = lambda p, x: (
                    p[0]*np.exp(-x*p[1]), 
                    p[2]*p[0]/p[1]*(np.exp(-x*0)-np.exp(-x*p[1])) + p[3]*np.exp(-x*0)
                    )

        return self._fit(fitfunc, p0, columns)
        self.fitfunc = fitfunc  #XXX hack
        errfunc = lambda p, x, y, xerr: (np.hstack(fitfunc(p, x))-np.hstack((y[0,:], y[1,:])))/\
                (np.hstack((xerr[0,:], xerr[1,:]))+0.02)

        args=(self.time, self.data_mean, self.data_std)
        return self.fitter(p0, errfunc, args)


    def fit_change(self, p0=[260, 10., 0.0], columns=[0,1], mask=slice(None), bounds=None, loss=0.):
        # p = [N1(0), r1, N2(0)]
        fitfunc = lambda p, x: (
                np.exp(-x*p[1])*p[0],
                np.exp(-x*loss)*( p[2] + p[0]*p[1]/(loss-p[1])*(np.exp(-x*(p[1]-loss))-1))
                )

        return self._fit(fitfunc, p0, columns, mask, bounds)

    def fit_change_disc(self, p0=[260, 10., 0.0, 1.0], columns=[0,1], mask=slice(None), bounds=None):
        # p = [N1(0), r1, N2(0), disc12]
        fitfunc = lambda p, x: (
                p[0]*np.exp(-x*p[1]),
                (p[0]*(np.exp(-x*0)-np.exp(-x*p[1])) + p[2]*1.)*p[3]
                )

        return self._fit(fitfunc, p0, columns, mask, bounds)


    def fit_H2O(self, p0=[260, 0.1, 0.01, 100, 0.1], t0_subtract=False):
        if t0_subtract:
            t0 = self.time[0]
        else:
            t0 = 0
        def fitfuncinc(p, x):
            return np.hstack((
                abs(p[0])*np.exp(-(x-t0)/abs(p[1])) + abs(p[3]), 
                abs(p[1]*p[0]/p[4])*(1-np.exp(-(x-t0)/abs(p[1]))) + abs(p[2])
                ))
        # assuming r1 = r2  i.e. sum is constant
        self.fitfuncinc = fitfuncinc  #XXX hack
        errfunc = lambda p, x, y, xerr: (fitfuncinc(p, x)-np.hstack((y[0,:], y[1,:])))/\
                (np.hstack((xerr[0,:], xerr[1,:]))+0.02)

        args=(self.time, self.data_mean, self.data_std)

        return self.fitter(p0, errfunc, args)


    def fit_equilib(self, p0=[260, 0.1, 100, 0.01], t0_subtract=False):
        if t0_subtract:
            t0 = self.time[0]
        else:
            t0 = 0
        def fitfuncinc(p, x):
            N10, r1, N20, r2 = p
            C2 = (N10 + N20)/(1+r1/r2)
            C1 = N10 - C2
            return np.hstack((
                C1*np.exp(-(x-t0)*(r1+r2)) + C2,
                -C1*np.exp(-(x-t0)*(r1+r2)) + C2*r1/r2
                ))
        # assuming r1 = r2  i.e. sum is constant
        self.fitfuncinc = fitfuncinc  #XXX hack
        def errfunc(p, x, y, xerr):
            return (fitfuncinc(p, x)-np.hstack((y[0,:], y[1,:])))/\
                (np.hstack((xerr[0,:], xerr[1,:]))+0.02)

        args=(self.time, self.data_mean, self.data_std)

        return self.fitter(p0, errfunc, args)


    def fit_H2O_disc(self, p0=[260, 0.1, 0.01, 100, 0.1, 1.], t0_subtract=False):
    #def fitOH(self, p0=[260, 1., 1., 0.01], full_output=False):
        #fitfuncinc = lambda p, x: p[0]*(1-np.exp(-x/p[1]))+p[2]
        if t0_subtract:
            t0 = self.time[0]
        else:
            t0 = 0
        def fitfuncinc(p, x):
            return np.hstack((
                abs(p[0])*np.exp(-(x-t0)/abs(p[1])) + abs(p[3]), 
                p[5]*( abs(p[1]*p[0]/p[4])*(1-np.exp(-(x-t0)/abs(p[1]))) + p[2] )
                ))
        # assuming r1 = r2  i.e. sum is constant
        self.fitfuncinc = fitfuncinc  #XXX hack
        errfunc = lambda p, x, y, xerr: (fitfuncinc(p, x)-np.hstack((y[0,:], y[1,:])))/\
                (np.hstack((xerr[0,:], xerr[1,:]))+0.02)

        args=(self.time, self.data_mean, self.data_std)

        return self.fitter(p0, errfunc, args)


    def fit_D_H2(self, p0=[2400.0, 3, 0.4, .2, 1.2, 70], column=0):
        from scipy.integrate import odeint
        def fitfunc(p, x):
            disc = p[4]
            eqn = lambda y, x: [\
                    -p[1]*y[0] + p[2]*y[1],\
                    (p[1]*y[0] - (p[2]+p[3])*y[1])/disc\
                    ]
            y0 = [p[0], p[5]]
            t = np.r_[0, x]
            y = odeint(eqn, y0, t)
            return (y[1:,0], y[1:,1])
        self.fitfunc = fitfunc  #XXX hack
        def errfunc( p, x, y, xerr):
            return (np.hstack(fitfunc(p, x))-np.hstack((y[0,:], y[1,:])))/\
                (np.hstack((xerr[0,:], xerr[1,:]))+0.02)
        args=(self.time, self.data_mean, self.data_std)

        return self.fitter(p0, errfunc, args)


    def fit_CO2(self, p0=[150, 0.1, 0.05, 0.1, 0.1]):
        from scipy.integrate import odeint
        def fitfunc(p, x):
            eqn = lambda y, x: [\
                    -(p[1]+p[2])*y[0],\
                    p[1]*y[0],\
                    p[2]*y[0],\
                    ]
            y0 = [p[0], p[3], p[4]]
            t = np.r_[0, x]
            y = odeint(eqn, y0, t)
            return (y[1:,0], y[1:,1], y[1:,2])
        self.fitfunc = fitfunc  #XXX hack
        def errfunc( p, x, y, xerr):
            return (np.hstack(fitfunc(p, x))-np.hstack((y[0,:], y[1,:], y[2,:])))/\
                    (np.hstack((xerr[0,:], xerr[1,:], xerr[2,:]))+0.02)
        args=(self.time, self.data_mean, self.data_std)

        return self.fitter(p0, errfunc, args)



    def fit_CO2_coupled(self, rate2, p0=[150, 0.1, 0.05, 0.1, 0.1, 150, 0.1, 0.05, 0.1], columns=[0, 1, 2]):
        from scipy.integrate import odeint
        def fitfunc(p, x):
            eqn = lambda y, x: [\
                    -(p[1]+p[2])*y[0],\
                    p[1]*y[0],\
                    p[2]*y[0],\
                    ]
            y0 = [p[0], p[4], p[3]]
            t = np.r_[0, x]
            y = odeint(eqn, y0, t)
            return (y[1:,0], y[1:,1], y[1:,2])
        self.fitfunc = fitfunc  #XXX hack
        rate2.fitfunc = fitfunc
        def errfunc( p, x, y, xerr, x2, y2, xerr2):
            p1 = p[:5]
            err1 = (np.vstack(fitfunc(p1, x))-y[columns,:])/\
                    (xerr[columns,:]+0.02)
            p1 = list(p[5:])+[p[4]]
            err2 = (np.vstack(fitfunc(p1, x2))-y2[columns,:])/\
                    (xerr2[columns,:]+0.02)
            return np.hstack((err1.ravel(), err2.ravel()))
        args=(self.time, self.data_mean, self.data_std, rate2.time, rate2.data_mean, rate2.data_std)

        return self.fitter(p0, errfunc, args)

    def fit_change_coupled(self, rate2, p0=[150, 0.1, 0.05, 150, 0.1], colmap=[0,1], loss=None,
            bounds=[
                (0, 1e100, 1e2),\
                (0, 1e100, 1e5),\
                (0, 1e100, 1e4),\
                (0, 1e100, 1e2),\
                (0, 1e100, 1e2)
                        ]\
            ):
        if loss == None:
            fitfunc = lambda p, x: (
                    p[0]*np.exp(-x*p[1]), 
                    p[0]*(np.exp(-x*0)-np.exp(-x*p[1])) + p[2]*1.
                    )
        else:
            fitfunc = lambda p, x: (
                    np.exp(-x*p[1])*p[0], 
                    np.exp(-x*loss)*( p[2] + p[0]*p[1]/(loss-p[1])*(np.exp(-x*(p[1]-p[2]))-1))
                    )
        """
        fitfunc = lambda p, x: (
                p[0]*np.exp(-x*p[1]), 
                p[0]*(np.exp(-x*0)-np.exp(-x*p[1])) + p[2]*1.
                )
        """
        self.fitfunc = fitfunc  #XXX hack
        rate2.fitfunc = fitfunc
        def errfunc( p, x, y, xerr, x2, y2, xerr2):
            p1 = p[:3]
            err1 = (np.hstack(fitfunc(p1, x))-np.hstack((y[colmap[0],:], y[colmap[1],:])))/\
                (np.hstack((xerr[colmap[0],:], xerr[colmap[1],:]))+0.1)
            p1 = list(p[3:])+[p[2]]
            err2 = (np.hstack(fitfunc(p1, x2))-np.hstack((y2[colmap[0],:], y2[colmap[1],:])))/\
                (np.hstack((xerr2[colmap[0],:], xerr2[colmap[1],:]))+0.1)
            return np.hstack((err1, err2))
        args=(self.time, self.data_mean, self.data_std, rate2.time, rate2.data_mean, rate2.data_std)

        return self.fitter(p0, errfunc, args, bounds=bounds)


    def linfit(self, p0=[60.0, 1.0], column=0):
        fitfunc = lambda p, x: p[0]*(1.0 - x*p[1])
        fitfunc = lambda p, x: p[0] - x*p[1]
        self.linfitfunc = fitfunc  #XXX hack
        errfunc = lambda p, x, y, xerr: (fitfunc(p, x)-y)/(xerr+0.2)
        args = (self.time, np.log(self.data_mean[column,:]), self.data_std[column,:]/self.data_mean[column,:])
        return self.fitter(p0, errfunc, args)


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


    def _fit(self, fitfunc, p0, columns, mask=slice(None), t0=0., boundweight=1e3):
        self.fitfunc = fitfunc # store the fitfunc for later
        self.fitcolumns = columns
        self.fit_t0 = t0
        self.fitmask = mask

        if type(p0) is dict:
            p0 = dict2Params(p0)
        else:
            p0 = p0.copy()

        # use custom implementation of bounded parameters, which can
        # estimate errors even if the parameter is "forced" outside the interval
        # idea: detect if fitted parameter is close to boundary, then make it fixed...
        bounds = {}
        for key, p in p0.items():
            if np.any(np.isfinite([p.min, p.max])):
                bounds[key] = (p.min, p.max, boundweight)
                p.set(min=-np.inf, max=np.inf)

        def errfunc( p, x, y, xerr, norm=1):
            sigma_min = 0.01
            res = (fitfunc(p, x)*norm-y[columns,:])/\
                (xerr[columns,:] + sigma_min)
            return res.ravel()

        def errfunc_multi(p, rates):
            err = []
            for i, rate in enumerate(rates):
                norm = p["n%d"%i].value if (i>0 and self.normalized) else 1
                err.append(errfunc(p, rate.time[mask] - t0, rate.data_mean[:,mask], rate.data_std[:,mask], norm))

            penalty = [weight*(np.fmax(lo-p[key], 0) + np.fmax(0, p[key]-hi))\
                        for key, (lo, hi, weight) in bounds.items()]
            err.append(penalty)

            #print("err = ", np.sum(np.hstack(err)**2))
            return np.hstack(err)

        # If needed, the normalization factors are appended to the list of
        # fitting parameters
        if self.normalized:
            for i in range(1,len(self.rates)): p0.add("n%d"%i, value=1)

        self.fitparam, pval, result = fitter(p0, errfunc_multi, (self.rates,))

        # extract the normalization factors
        if self.normalized:
            for i in range(1,len(self.rates)): self.norms[i] = self.fitparam["n%d"%i].value

        return self.fitparam, pval, result


    def fit_decay(self, p0, columns=[0], mask=slice(None), t0=0.):

        def fitfunc (p, x):
            N0 = p["N0"].value
            rate = p["rate"].value

            return (N0*np.exp(-x*rate), )

        return self._fit(fitfunc, p0, columns, mask, t0)


    def fit_change(self, p0, columns=[0,1], mask=slice(None), bounds=None, t0=0., loss=0.):

        def fitfunc(p, x):
            N0 = p["N0"].value
            N1 = p["N1"].value
            rate = p["rate"].value
            return (
                np.exp(-x*rate)*N0,
                np.exp(-x*loss)*(N1 + N0*rate/(loss-rate)*(np.exp(-x*(rate-loss))-1))
                )

        return self._fit(fitfunc, p0, columns, mask, t0)


    def fit_change_channel(self, p0, columns=[0,1], mask=slice(None), bounds=None, t0=0.):

        def fitfunc(p, x):
            N0 = p["N0"].value
            N1 = p["N1"].value
            rate = p["rate"].value
            bratio = p["bratio"].value
            return (
                np.exp(-x*rate)*N0,
                N0*bratio*(1-np.exp(-x*rate)) + N1
                )

        return self._fit(fitfunc, p0, columns, mask, t0)

    def fit_NH(self, p0={
        "N": 100.,          "NH": 10.,
        "NH2": 1.,          "NH3": 1.,
        "H3": 10.,          "N15":10.,
        "rNH":1.,          "rNH2":10.,
        "rH3":1,            "rNH3":10,
        "rH3d":1},\
            columns=[0,1,2,3,4], mask=slice(None), t0=0):

        from scipy.integrate import odeint
        specnames = ["N", "NH", "NH2", "NH3", "H3", "N15"]
        ratenames = ["rNH", "rNH2", "rH3", "rNH3", "rH3d"]

        p0 = dict2Params(p0)
        for key in p0: p0[key].set(min=0)
        def fitfunc(p, x):
            rNH, rNH2, rH3, rNH3, rH3d = [p[name].value for name in ratenames]
            N, NH, NH2, NH3, H3, N15 = range(6)
            eqn = lambda y, x: [\
                    # N+
                    -rNH*y[N],\
                    # NH+
                    -(rNH2 + rH3)*y[NH] + rNH*y[N],\
                    # NH2+
                    rNH2*y[NH] - rNH3*y[NH2],\
                    # NH3+
                    rNH3*y[NH2],\
                    # H3+
                    rH3*y[NH] - rH3d*y[H3],\
                    # 15N+
                    -rNH*y[N15],
                    ]
            y0 = [p[name].value for name in specnames]
            t = np.r_[0, x]
            y = odeint(eqn, y0, t, mxstep=10000)
            res = y[1:,:5]
            res[:,1] += y[1:,N15] # add the relaxed and excite NH3+
            return res.T

        return self._fit(fitfunc, p0, columns, mask, t0)


    def fit_NHn_long(self, p0={
        "N": 100.,          "NH": 10.,
        "NH2": 1.,          "NH3": 1.,
        "NH4": .1,          "H3": 10.,
        "rNH":1.,           "rNH2":10.,
        "rH3":1,            "rNH3":10,
        "rNH4":1,           "rH3d":1,
        "rNH3rel":1,        "rNH4exc":10},\
            columns=[0,1,2,3,4,5], mask=slice(None), t0=0, discrimination=False, NH3loss=0.05, H3disc=1.):
        from scipy.integrate import odeint
        if discrimination is True:
            raise NotImplementedError("discrimination fit in fit_NHn_long not implemented")

        p0 = dict2Params(p0)
        for key in p0: p0[key].set(min=0)

        def fitfunc(p, x):
            N, NH, NH2, NH3, NH4, H3, NH3e = range(7)
            eqn = lambda y, x: [\
                    # N+
                    -p["rNH"]*y[N],\
                    # NH+
                    p["rNH"]*y[N] - p["rH3"]*y[NH] - p["rNH2"]*y[NH],\
                    # NH2+
                    p["rNH2"]*y[NH] - p["rNH3"]*y[NH2],\
                    # NH3+ relaxed
                    p["rNH3rel"]*y[NH3e] - p["rNH4"]*y[NH3] - NH3loss*y[NH3],\
                    # NH4+
                    p["rNH4"]*y[NH3] + p["rNH4exc"]*y[NH3e],\
                    # H3+
                    p["rH3"]*y[NH] - p["rH3d"]*y[H3],\
                    # excited NH3+
                    p["rNH3"]*y[NH2] - p["rNH3rel"]*y[NH3e] - p["rNH4exc"]*y[NH3e],\
                    ]
            y0 = [p["N"], p["NH"], p["NH2"], p["NH3"], p["NH4"], p["H3"], 0.]
            t = np.r_[0, x]
            y = odeint(eqn, y0, t, mxstep=10000)
            res = y[1:,[0,1,2,3,4,5]]
            res[:,3] += y[1:,NH3e] # sum the relaxed and excited NH3+
            #res *= np.array([1] + list(disc))
            res[:,H3] *= H3disc
            return res.T

        return self._fit(fitfunc, p0, columns, mask, t0)


    def fit_NHn_short(self, p0={
        "N":400.,       "NH":.1,
        "NH2":.1,       "NH3":.1,
        "H3":.1,        "rNH":10.,
        "rNH2":100.,    "rNH3":100.,
        "rNH4":10.,     "rH3":10,
        "rH3d":10.
        },\
            columns=[0,1,2,3,5], mask=slice(None), t0=0, H3disc = 1.):
        from scipy.integrate import odeint

        #[400, 10., 100., 100., 10., 10., 10., .1, .1, .1, .1]
        #pnames = ["NH+ rate", "NH2+ rate", "NH3+ exc rate", "NH4+ rate", "H3+ rate", "-H3+ rate",\
        #                "NH+(0)", "NH2+(0)", "NH3+(0)", "H3+(0)"]
        p0 = dict2Params(p0)
        for key in p0: p0[key].set(min=0)

        def fitfunc(p, x):
            N, NH, NH2, NH3, H3, NH4 = range(6)
            eqn = lambda y, x: [\
                    # N+
                    -p["rNH"]*y[N],\
                    # NH+
                    p["rNH"]*y[N] - p["rH3"]*y[NH] - p["rNH2"]*y[NH],\
                    # NH2+
                    p["rNH2"]*y[NH] - p["rNH3"]*y[NH2],\
                    # NH3+ relaxed
                    p["rNH3"]*y[NH2] - p["rNH4"]*y[NH3],\
                    # H3+
                    p["rH3"]*y[NH] - p["rH3d"]*y[H3],\
                    ]
            y0 = [p["N"], p["NH"], p["NH2"], p["NH3"], p["H3"]]
            t = np.r_[0, x]
            y = odeint(eqn, y0, t, mxstep=10000)
            res = y[1:,[0,1,2,3,4]]
            res[:,H3] *= H3disc
            return res.T

        return self._fit(fitfunc, p0, columns, mask, t0)

    def fit_NHn_nodisc_short(self, p0=[400, 10., 100., 100., 10., 10., 10., .1, .1, .1, .1],\
            columns=[0,1,2,3,5], mask=slice(None), t0=0, H3disc = 1.):
        from scipy.integrate import odeint

        def fitfunc(p, x):

            if len(p)>11: raise RuntimeError("fitfunc: too many parameters" + str(p))
            #print(", ".join("%7.2f" % ii for ii in p))
            N, NH, NH2, NH3, H3, NH4 = range(6)
            eqn = lambda y, x: [\
                    # N+
                    -p[1]*y[N],\
                    # NH+
                    (p[1]*y[N] - p[2]*y[NH] - p[5]*y[NH]),\
                    # NH2+
                    (p[2]*y[NH] - p[3]*y[NH2]),\
                    # NH3+ relaxed
                    (p[3]*y[NH2] - p[4]*y[NH3]),\
                    # H3+
                    (p[5]*y[NH] - p[6]*y[H3]),\
                    ]
            y0 = [p[0], p[7], p[8], p[9], p[10]]
            t = np.r_[0, x]
            y = odeint(eqn, y0, t, mxstep=10000)
            res = y[1:,[0,1,2,3,4]]
            res[:,H3] *= H3disc
            return res.T


        bounds = [(0, 1e9, 1e3)]*len(p0)

        return self._fit(fitfunc, p0, columns, mask, bounds, t0)

    def fitOH(self, p0=[10., 1, 0.01], nions=100., OH_loss=False, OH_injection=False):
    #def fitOH(self, p0=[260, 1., 1., 0.01], full_output=False):
        #fitfuncinc = lambda p, x: p[0]*(1-np.exp(-x/p[1]))+p[2]
        nrates = len(self.rates)
        if OH_loss:
            if OH_injection:
                if len(p0)==3:
                    p0 = p0 + [0]
                fitfunc = lambda p, x: (
                        p[0]*np.exp(-x*p[1]), 
                        p[2]*p[0]/(p[1]-p[3])*(np.exp(-x*(p[3]))-np.exp(-x*p[1])) +\
                        (p[4])*np.exp(-x*(p[3]))
                        )
            else:
                fitfunc = lambda p, x: (
                        p[0]*np.exp(-x*p[1]), 
                        p[2]*p[0]/(p[1]-(p[3]))*(np.exp(-x*(p[3]))-np.exp(-x*p[1])) +\
                        0.*np.exp(-x*(p[3]))
                        )
        else:
            fitfunc = lambda p, x: (
                    p[0]*np.exp(-x*p[1]), 
                    p[2]*p[0]/p[1]*(np.exp(-x*0)-np.exp(-x*p[1])) + p[3]*np.exp(-x*0)
                    )
        self.fitfunc = fitfunc  #XXX hack
        errfunc = lambda p, x, y, xerr: (np.hstack(fitfunc(p, x))-np.hstack((y[0,:], y[1,:])))/\
                (np.hstack((xerr[0,:], xerr[1,:]))+0.02)


        def errfunc_multi(p, rates):
            # p[0] is normalization factor (# of ions)
            err = []
            for i in range(nrates):
                pi = list([p[i]])+list(p[nrates:])
                err.append(errfunc(pi, rates[i].time, rates[i].data_mean, rates[i].data_std))
            return np.hstack(err)


        p0 = [nrates]*len(self.rates) + p0
        return self.fitter(p0, errfunc_multi, (self.rates,))


    def fit_D_H2(self, p0=[3, 0.4, .2, 1.2, 70], nions=2400, column=0):
        print(p0)
        from scipy.integrate import odeint
        nrates = len(self.rates)
        def fitfunc(p, x):
            disc = p[4]
            eqn = lambda y, x: [\
                    -p[1]*y[0] + p[2]*y[1],\
                    (p[1]*y[0] - (p[2]+p[3])*y[1])/disc\
                    ]
            y0 = [p[0], p[5]]
            t = np.r_[0, x]
            y = odeint(eqn, y0, t)
            return (y[1:,0], y[1:,1])
        self.fitfunc = fitfunc  #XXX hack
        def errfunc( p, x, y, xerr):
            return (np.hstack(fitfunc(p, x))-np.hstack((y[0,:], y[1,:])))/\
                (np.hstack((xerr[0,:], xerr[1,:]))+0.02)

        def errfunc_multi(p, rates):
            # p[0] is normalization factor (# of ions)
            err = []
            for i in range(nrates):
                pi = list([p[i]])+list(p[nrates:])
                err.append(errfunc(pi, rates[i].time, rates[i].data_mean, rates[i].data_std))
            #print "err = ", np.sum(np.hstack(err)**2)
            return np.hstack(err)


        p0 = [nions]*nrates + list(p0)
        return self.fitter(p0, errfunc_multi, (self.rates,))
