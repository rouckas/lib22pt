import numpy as np

# 2013-10-31 Added MultiRate class, simplified fitting methods, removed full_output parameter
# 2014-12-18 Add loading of Frequency, Integration time and Iterations, calculate lower
#            bound on errors from Poisson distribution
# 2015-01-28 Simplified fitting again. Needs more work
# 2015-03-02 Added functions for number density calculation

def concentrationD2(temp, Paml22PT, PamlSIS=0., f_D2_22PT=56):
    k_B = 1.3806488e-23
    return (Paml22PT*f_D2_22PT + PamlSIS*1.4)/k_B/np.sqrt(300*temp)/10000

def concentrationH2(temp, Paml22PT, PamlSIS, f_H2_22PT=38.2):
    k_B = 1.3806488e-23
    return (Paml22PT*f_H2_22PT + PamlSIS*1.4)/k_B/np.sqrt(300*temp)/10000

def concentrationHe(temp, Paml22PT, PamlSIS, f_He_22PT=140.):
    k_B = 1.3806488e-23
    #f_22PT = 140.      # calibration from 2013-02-22
    f_SIS = 5.          # rough estimate, mostly negligible
    return (Paml22PT*f_He_22PT + PamlSIS*f_SIS)/k_B/np.sqrt(300*temp)/10000

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

            if toks[0] == "Time":
                state = 1
                time = np.zeros(npoints)
                data = np.zeros((nions, npoints, self.niter))
                if len(ioniter) != nions:
                    print("Corrupt file", fname, "Iterations for all species not recorded")
                    while len(ioniter) < nions:
                        ioniter.append(ioniter[-1])
                continue

            if state == 1:
                try:
                    newtime = float(toks[0])
                except ValueError:
                    state = 2
                else:
                    if pointno==len(time):
                        print(("Corrupt file " + fname + " trying to guess number of points"))
                        npoints += 1
                        time.resize(npoints)
                        data.resize((nions, npoints, self.niter)) 
                    time[pointno] = newtime
                    pointno += 1

            if state == 2:
                if toks[0] == "Iteration":
                    if pointno != npoints:
                        print(("Corrupt file " + fname + " trying to guess number of points"))
                        #msg = "Corrupt file: " + fname
                        #raise IOError(msg)
                        npoints = pointno
                        time.resize(npoints)
                        data.resize((nions, npoints, self.niter)) 

                    iterno = int(toks[1])-1
                    if iterno+1 > self.niter:
                        print(("Corrupt file " + fname + " trying to guess number of iterations"))
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
                #XXX works for smaller niter only
                #raise IOError(msg)
            data = data[:,:,:iterno+1]
        
        #print skip_iter, np.shape(skip_iter)
        if len(skip_iter)!=0:
            skip_iter = np.array(skip_iter)
            indices = np.ones(self.niter, dtype=bool)
            indices[skip_iter] = False
            data = data[:,:,indices]

        # XXX frequency is sometimes wrong in the files
        self.total_iterations = ioniter[:,None]*integration*frequency*self.niter

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
    def fitter(self, p0, errfunc, args, bounds=None):
        if bounds==None:
            _errfunc = errfunc
        else:
            def _errfunc(p, *args):
                penalty = [weight*(np.fmax(lo-pp, 0) + np.fmax(0, pp-hi))\
                        for pp, (lo, hi, weight) in zip(p, bounds)]
                residuals = errfunc(p, *args)
                #print("residuals:", np.sum(penalty**2), np.sum(residuals**2))
                return np.hstack((residuals, penalty))

        from scipy.optimize import leastsq
        p, cov_p, info, mesg, ier \
            = leastsq(_errfunc, p0, args=args, full_output=1, factor=0.9, maxfev=2000)
        
        if ier not in [1, 2, 3, 4]:
            msg = "ier"+str(ier) + " Optimal parameters not found: " + mesg
            raise RuntimeError(msg)
        if cov_p is None:
            return p, 1./np.zeros_like(p), 0.
            raise RuntimeError("Optimal parameters not found: Covariance is None")
        if any(np.diag(cov_p) < 0):
            raise RuntimeError("Optimal parameters not found: negative variance")
        
        chisq = np.dot(info["fvec"], info["fvec"])
        dof = len(info["fvec"]) - len(p)
        sigma = np.array([np.sqrt(cov_p[i,i])*np.sqrt(chisq/dof) for i in range(len(p))])

        Q = chisq/dof
        from scipy.stats import chi2
        chi = chi2.cdf(Q, dof)
        #if chi > 0.5: pval = (1-chi)*2
        #else: pval = chi*2
        pval = chi
        return p, sigma, pval

    def _fit(self, fitfunc, p0, columns, mask=slice(None)):
        self.fitfunc = fitfunc # store the fitfunc for later
        def errfunc( p, x, y, xerr):
            sigma_min = 0.02
            retval = (fitfunc(p, x)-y[columns,:])/\
                (xerr[columns,:] + sigma_min)
            print(p, np.sum(retval**2))
            return retval.ravel()
        args=(self.time[mask], self.data_mean[:,mask], self.data_std[:,mask])
        return self.fitter(p0, errfunc, args)


    def fit_decay(self, p0=[60.0, .1], columns=0, mask=slice(None)):
        fitfunc = lambda p, x: p[0]*np.exp(-x*p[1])
        return self._fit(fitfunc, p0, columns, mask)

    def fit_ode_mpmath(self, p0=[60.0, .1], columns=0):
        from mpmath import odefun
        def fitfunc(p, x):
            eqn = lambda x, y: -p[1]*y
            y0 = p[0]
            f = odefun(eqn, 0, y0)
            g = np.vectorize(lambda x: float(f(x)))
            return g(x)

        return self._fit(fitfunc, p0, columns)


    def fit_ode_scipy(self, p0=[60.0, .1], columns=0):
        from scipy.integrate import odeint
        def fitfunc(p, x):
            eqn = lambda y, x: -p[1]*y
            y0 = p[0]
            t = np.r_[0., x]
            y = odeint(eqn, y0, t)
            return y[1:,0]
        return self._fit(fitfunc, p0, columns)


    def fit_inc(self, p0=[1.0, .01, 0.99], columns=1):
        #fitfuncinc = lambda p, x: p[0]*(1-np.exp(-x/p[1]))+p[2]
        fitfunc = lambda p, x: -abs(p[0])*np.exp(-x/abs(p[1]))+abs(p[2])
        return self._fit(fitfunc, p0, columns)
    
    def fit_equilib(self, p0=[70.0, .1, 1], columns=0):
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


    def fit_change(self, p0=[260, 10., 0.0], columns=[0,1]):
        # p = [N1(0), r1, N2(0)]
        fitfunc = lambda p, x: (
                p[0]*np.exp(-x*p[1]), 
                p[0]*(np.exp(-x*0)-np.exp(-x*p[1])) + p[2]*1.
                )

        return self._fit(fitfunc, p0, columns)


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



    def fit_CO2_coupled(self, rate2, p0=[150, 0.1, 0.05, 0.1, 0.1, 150, 0.1, 0.05, 0.1]):
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
            err1 = (np.hstack(fitfunc(p1, x))-np.hstack((y[0,:], y[1,:], y[2,:])))/\
                    (np.hstack((xerr[0,:], xerr[1,:], xerr[2,:]))+0.02)
            p1 = list(p[5:])+[p[4]]
            err2 = (np.hstack(fitfunc(p1, x2))-np.hstack((y2[0,:], y2[1,:], y2[2,:])))/\
                    (np.hstack((xerr2[0,:], xerr2[1,:], xerr2[2,:]))+0.02)
            return np.hstack((err1, err2))
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


class MultiRate(Rate):
    def __init__(self, fnames, directory=""):
        self.rates = []
        for fname in fnames:
            self.rates.append(Rate(directory+fname, full_data=True))

    def fit(self, p0=[.1], nions=60., column=0):

        nrates = len(self.rates)
        fitfunc = lambda p, x: p[0]*np.exp(-x*p[1])
        self.fitfunc = fitfunc  #XXX hack
        errfunc = lambda p, x, y, xerr: (fitfunc(p, x)-y)/(xerr+0.02)

        def errfunc_multi(p, rates):
            # p[0] is normalization factor (# of ions)
            err = []
            for i in range(nrates):
                pi = list([p[i]])+list(p[nrates:])
                err.append(errfunc(pi, rates[i].time, rates[i].data_mean[column,:], rates[i].data_std[column,:]))
            return np.hstack(err)


        p0 = [nions]*nrates + p0
        return self.fitter(p0, errfunc_multi, (self.rates,))


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



#XXX attention, new fitting methods return rates instead of taus!!!
class Labbook:
    def __init__(self, fname, datename, direc="../Rate", fmt=[("id", np.str_, 2), ("T", np.float_), ("shut", np.str_, 1), ("Ubar", np.int_), ("gid", np.int_)], full_data=True):
        self.labbook = np.loadtxt(fname, fmt, comments="#")
        self.datename = datename
        self.full_data = full_data
        self.opened = (self.labbook["shut"] == "O")
        self.laser = (self.labbook["shut"] == "L")
        self.bg = (self.labbook["shut"] == "C")
        self.direc = direc

    def fit(self):
        data = None
        index = None
        for index in range(len(self.labbook)):
            i = self.labbook[index]
            ratefile = self.direc + "/r" + self.datename + "_"+i[0]+".dat"
            try:
                rate = Rate(ratefile, self.full_data)
            except (IOError):
                print("Unreadable file " + ratefile)
                continue
                #rate.cut3sigma()
                #rate.poisson_test1()

            try:
                tau, tauerr = rate.fit()
            except (RuntimeError):
                if i["shut"] == "C":
                    try:
                        tau, tauerr = rate.linfit()
                    except (RuntimeError):
                        pass
                else:
                    continue

            if abs(tauerr) > abs(tau):
                try:
                    tau, tauerr = rate.linfit()
                except (RuntimeError):
                    pass

            line = np.r_[tau, tauerr]
            if data is not None:
                data = np.vstack((data, line))
                indices = np.vstack((indices, index))
            else:
                data = line
                indices = index
            print(i[0], tau)

        if data is None:
            raise RuntimeError("No data fitted in date %s" % (self.datename,))

        rates = 1.0/data[:,0]
        rateerrs = data[:,1]/data[:,0]**2

        #drop negative rates
        #rates[rates<0] = 0.0

        ii = self.labbook[indices]["T"][:,0]
        self.data = np.vstack((ii, rates, rateerrs)).T
        self.labbook = self.labbook[indices]
        self.opened = self.opened[indices][:,0]
        self.laser = self.laser[indices][:,0]
        self.bg = self.bg[indices][:,0]
        return self.data, self.labbook

    def bg_substract(self):
        opened = (self.labbook["shut"] == "O")[:,0]
        for i in range(len(self.labbook)):
            if not opened[i]: continue
            gid = self.labbook["gid"][i]
            indices = (self.labbook["gid"] == gid)[:,0] & ~opened
            if gid == 0 or len(self.data[indices,1]) == 0: continue
            bgs = np.mean(self.data[indices,1])
            bgerrs = np.sqrt(np.sum(self.data[indices,2]**2)/len(self.data[indices,2]))
            #bgerrs = np.std(data[indices,2])/np.sqrt(len(data[indices,2]))
            self.data[i, 1] -= bgs
            self.data[i, 2] = np.sqrt(self.data[i,2]**2 + bgerrs**2)
        return self.data, self.labbook

    def merge_points(self, mask=None):
        if mask==None: mask = np.ones(len(self.opened), dtype=bool)
        # beware, std deviation is returned instead of mean error
        gids = set(self.labbook["gid"][:,0])
        #opened = (self.labbook["shut"] == "O")[:,0]
        data2 = self.data[0,:]
        labbook2 = self.labbook[0]
        for gid in gids:
            Oindices = (self.labbook["gid"] == gid)[:,0] & self.opened & mask
            Cindices = (self.labbook["gid"] == gid)[:,0] & self.bg & mask
            Lindices = (self.labbook["gid"] == gid)[:,0] & self.laser & mask
            for indices in [Oindices, Cindices, Lindices]:
                litem = self.labbook[indices]
                if len(litem) > 0:
                    item = np.mean(self.data[indices,:], axis=0)
                    item[2] = np.std(self.data[indices,1])
                    data2 = np.vstack((data2, item))
                    labbook2 = np.vstack((labbook2, litem[0]))

        #data2 = data2[~np.isnan(data2[:,0]),:]
        labbook2 = labbook2[1:]
        data2 = data2[1:]
        self.labbook_merged = labbook2
        self.data_merged = data2
        self.opened_merged = (labbook2["shut"] == "O")[:,0]
        self.laser_merged = (labbook2["shut"] == "L")[:,0]
        self.bg_merged = (labbook2["shut"] == "C")[:,0]

        #backward compatibility:
        self.opened2 = self.opened_merged
        return data2, labbook2

    def print_summary(self):
        for i in range(len(self.data)):
            print((self.labbook["id"][i], self.labbook["shut"][i], self.data[i]))

    def mask(self, skippts = []):
        mask = np.ones(len(self.opened), dtype=bool)
        for i in skippts:
            mask[(self.labbook["id"] == "%.2d" % (i,))[:,0]] = False
        return mask

    def normalize_pressure(self):
        self.data[self.opened,1] = \
                self.data[self.opened,1]/self.labbook["Ubar"][self.opened][:,0]*1050

    def data_process(self, normalize_pressure=True, skippts=[]):
        self.fit()
        self.bg_substract()
        if normalize_pressure: self.normalize_pressure()
        return self.data, self.opened, self.mask(skippts)



def decimate(data, bins):
    averages = np.zeros((len(bins)-1, 5))
    for i in range(len(bins)-1):
        indices = (data[:,0] >= bins[i] ) & (data[:,0] < bins[i+1])
        if np.any(indices): averages[i,4] = 1
        else: continue
        subset = data[indices]
        from avg import w_avg_std

        averages[i,0], averages[i,2], dum = w_avg_std(subset[:,0], 1/subset[:,2]**2)
        averages[i,1], averages[i,3], dum = w_avg_std(subset[:,1], 1/subset[:,2]**2)
        if len(subset[:,0])==1:
            # give lower weight to measurements with single sample
            averages[i,2] = 0.
            averages[i,3] = subset[0,2]*2.
    return averages


def stitch(avg1, avg2):
    def distance(p, data_avg, averages):
        overlap = (avg1[:,4]>0) & (avg2[:,4]>0)
        dist = avg1[overlap,1]*p[0] - avg2[overlap,1]
        var = np.sqrt((avg1[overlap,3]*p[0])**2 + avg2[overlap,3]**2)
        return dist/var

    p0 = [1.0]
    from scipy.optimize import leastsq
    p, cov_p, info, mesg, ier \
        = leastsq(distance, p0, args=(avg1, avg2), full_output=1, factor=0.1)
    
    if ier not in [1, 2, 3, 4] or cov_p is None:
        msg = "Optimal parameters not found: " + mesg
        raise RuntimeError(msg)
    if any(np.diag(cov_p) < 0):
        raise RuntimeError("Optimal parameters not found: negative variance")
    
    p = [p]
    chisq = np.dot(info["fvec"], info["fvec"])
    dof = len(info["fvec"]) - len(p)
    sigma = np.array([np.sqrt(cov_p[i,i])*np.sqrt(chisq/dof) for i in range(len(p))])

    Q = chisq/dof
    from scipy.stats import chi2
    chi = chi2.cdf(Q, dof)
    #if chi > 0.5: pval = (1-chi)*2
    #else: pval = chi*2
    pval = chi
    return p, sigma, pval


def stitch_old(fname, averages, datasets, labbooks, bins, full_output=False):
    prefix, skippts, T, scalefact, col = datasets[fname]
    LB = labbooks[fname]
    mask = LB.mask(skippts)
    data = LB.data[LB.opened & mask]
    data[:,0] = (data[:,0]+T)/2.0

    data_avg = decimate(data, bins)

    def distance(p, data_avg, averages):
        overlap = (data_avg[:,1]>0) & (averages[:,1]>0)
        dist = data_avg[overlap,1]*p[0] - averages[overlap,1]
        return dist

    p0 = [1.0]
    from scipy.optimize import leastsq
    p, cov_p, info, mesg, ier \
        = leastsq(distance, p0, args=(data_avg, averages), full_output=1)
    
    if ier not in [1, 2, 3, 4] or cov_p is None:
        msg = "Optimal parameters not found: " + mesg
        raise RuntimeError(msg)
    if any(np.diag(cov_p) < 0):
        raise RuntimeError("Optimal parameters not found: negative variance")
    
    p = [p]
    chisq = np.dot(info["fvec"], info["fvec"])
    dof = len(info["fvec"]) - len(p)
    sigma = np.array([np.sqrt(cov_p[i,i])*np.sqrt(chisq/dof) for i in range(len(p))])
    if full_output:
        Q = chisq/dof
        from scipy.stats import chi2
        chi = chi2.cdf(Q, dof)
        #if chi > 0.5: pval = (1-chi)*2
        #else: pval = chi*2
        pval = chi
        return p, sigma, pval
    else:
        return p[0], sigma[0]
