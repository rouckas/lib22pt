#!/usr/bin/python
import numpy as N

def weighted_mean(x, w=None, std=None, errtype="", dropnan=False, full_output=False):
    """
    Returns weighted mean and mean of sample standard deviation of
    samples with known standard deviations or weights.
    x - samples (ndarray)
    std - standard deviations of samples (ndarray)
    w - weights of samples (ndarray)
    errtype:
        - "sample_weights":
            This choice weights the points by fit errors and returns the estimated variance
            of the input data (thus accounting for e.g. pressure variations etc...)
            Roughly equivalent to the old add_errs choice

        - "mean_weights":
            This choice weights the points by fit errors and returns the estimated variance
            of the weighted mean (thus accounting for e.g. pressure variations etc...)

        - "mean_std":
            this choice assumes that err are the true uncertainties of the data and the variance
            of the mean is calculated accordingly. This underestimates uncertainties if _err does
            not include the full statistical error

        - "mean_std_scaled":
            This choice assumes that err are the true uncertainties of the data and the variance
            of the mean is calculated accordingly. The error is scaled by chi2 if the uncertainties
            appear to  be underestimated

        - "mean_std_scaled_additive":
            This choice assumes that there is an unknown constant uncertainty of the data to be
            added to the provided std error.
    """

    if w is None and std is not None:
        if "weights" in errtype:
            w = 1/std**2
    elif std is None and w is not None:
        if "std" in errtype:
            std = N.sqrt(1/w)
    else:
        raise ValueError("weighted_mean(): Either w or std has to be set")

    if errtype == "sample_weights":
        # this choice weights the points by fit errors and returns the estimated variance
        # of the input data (thus accounting for e.g. pressure variations etc...)
        # Roughly equivalent to the old add_errs choice
        mean, err, _, *full = w_avg_std(x, w, dropnan=dropnan)

    if errtype == "mean_weights":
        # this choice weights the points by fit errors and returns the estimated variance
        # of the weighted mean (thus accounting for e.g. pressure variations etc...)
        mean, _, err, *full = w_avg_std(x, w, dropnan=dropnan)

    if errtype == "mean_std":
        # this choice assumes that err are the true uncertainties of the data and the variance
        # of the mean is calculated accordingly. This underestimates uncertainties if _err does
        # not include the full statistical error
        mean, err, *full = wstd_avg_std(x, std, dropnan=dropnan, scale_error=False, full_output=full_output)

    if errtype == "mean_std_scaled":
        # this choice assumes that err are the true uncertainties of the data and the variance
        # of the mean is calculated accordingly. The error is scaled by chi2 if the uncertainties appear to
        # be underestimated
        mean, err, *full = wstd_avg_std(x, std, dropnan=dropnan, scale_error="multiplicative", full_output=full_output)

    if errtype == "mean_std_scaled_additive":
        # this choice assumes that err are the true uncertainties of the data and the variance
        # of the mean is calculated accordingly. Constant uncertainty is added to the error 
        # if the uncertainties appear to be underestimated
        mean, err, *full = wstd_avg_std(x, std, dropnan=dropnan, scale_error="additive", full_output=full_output)

    if errtype == "sample_std_scaled_additive":
        # this choice assumes that err are the true uncertainties of the data and the variance
        # of the mean is calculated accordingly. Constant uncertainty is added to the error 
        # if the uncertainties appear to be underestimated. The estimated constant uncertainty
        # is added quadratically to the mean error, so the resulting error is indicative
        # of the data reproducibility
        mean, err, S = wstd_avg_std(x, std, dropnan=dropnan, scale_error="additive", full_output=True)
        err = N.sqrt(err**2 + S**2)
        if full_output: full = (S,)
        else: full = ()

    return mean, err, *full

def wstd_avg_std(x, std, axis=0, dropnan=False, scale_error=False, full_output=False):
    """
    Returns weighted mean and mean standard deviation of samples with known
    standard deviations. Optimal weighting is used.
    x - samples (ndarray)
    w - standard deviations of samples (ndarray)
    scale_error - if "multiplicative", and the reduced chi square of the data is > 1,
        the std**2 are multiplied by S**2 = chi2/(N-1) so that chi2' = N-1
        This is the error scaling procedure according to Tanabashi et al., 2018
        (page 16 in https://doi.org/10.1103/PhysRevD.98.030001)
        This solves the issue of possible underestimation of error estimates

        if "additive", and the reduced chi square of the data is > 1, a constant S**2
        is added to std**2 so that chi2' = (N-1)
        The S**2 term is found by numerical root finding. This is applicable in situations,
        when the provided std errors only contain a part of the uncertainty and the total
        uncertainty is std**2 + S**2 (for example std are fit errors, and S are errors in
        experimental parameters)
    """

    if N.shape(x)[axis] < 2:
        return N.rollaxis(x, axis)[0], N.rollaxis(std, axis)[0]

    if dropnan:
        I = (N.isfinite(x) & N.isfinite(std))
        x = x[I]
        std = std[I]

    if len(x) < 1:
        return N.nan, N.nan, N.nan
    if len(x) < 2:
        return x[0], std[0]

    w = 1/std**2
    sum_w = N.sum(w, axis=axis)

    xm = N.sum(w*x, axis=axis)/sum_w
    var = 1/sum_w # this holds for w = 1/Var[x]

    S = None

    if scale_error == "multiplicative":
        if axis != 0:
            raise NotImplementedError("Error scaling in multidimensional arrays not implemented")

        chi2 = N.sum(w * (x - xm)**2)
        Nsampl = len(x)
        redchi = chi2 / (Nsampl-1)

        #delta0 = 3*N.sqrt(Nsampl*var)

        S = N.sqrt(chi2/(Nsampl-1))
        if S > 1:
            var *= S
        else:
            S = 1

    elif scale_error == "additive":
        if axis != 0:
            raise NotImplementedError("Error scaling in multidimensional arrays not implemented")

        chi2 = N.sum(w * (x - xm)**2)
        Nsampl = len(x)
        redchi = chi2 / (Nsampl-1)
        if redchi > 1:
            # errors appear to be underestimated
            # we replace std with sqrt(std**2 + S**2)
            # and solve for S from equation redchi(S) - 1 = 0

            # initial estimate of S
            # assuming constant std
            stdmean = N.mean(std)
            S0 = N.sqrt(stdmean**2*(redchi - 1))

            def eq(S):

                w = 1/(std**2 + S**2)
                sum_w = N.sum(w)
                xm = N.sum(w*x, axis=axis)/sum_w
                res = N.sum((x - xm)**2 * w) - (Nsampl-1)
                return res

            """
            def eqprime(S):
                w = 1/(std**2 + S**2)
                w2 = w**2
                sum_w = N.sum(w)
                sum_w2 = N.sum(w2)
                # can we neglect the variation of xm with S?
                xmprime = N.sum(w2*x)/sum_w2
                res = N.sum(-2*S*(x - xm)**2 * w2 + 2*w*(xm - x)*xmprime)
                #print("S=", S, "eqprime=", res)
                return res

            from scipy.optimize import newton
            #S, r = newton(eq, S0, eqprime, full_output=True)
            """
            # Newton's method converges very slowly, Brent's method is order of magnitude faster (limited testing)

            from scipy.optimize import brentq
            # find bracketing points. eq(0) = (redchi - 1)*(N-1) > 1
            # so we need to find the negative point (overestimate errors)
            while eq(S0) > 0: S0 *= 2

            S, r = brentq(eq, 0, S0, full_output=True)

            w = 1/(std**2 + S**2)
            sum_w = N.sum(w)
            xm = N.sum(w*x, axis=axis)/sum_w
            var = 1/sum_w

        else:
            S = 0


    if full_output:
        return xm, N.sqrt(var), S
    else:
        return xm, N.sqrt(var)

def w_avg_std(x, w, dropnan=False):
    """
    Returns weighted mean,
        sqrt(weighted unbiased variance) of the sample,
        sqrt(weighted mean variance) of the mean,
    which is unbiased standard deviation
    x - samples (ndarray)
    w - weights (ndarray)
    """

    if dropnan:
        I = (N.isfinite(x) & N.isfinite(w))
        x = x[I]
        w = w[I]

    if len(x) < 1:
        return N.nan, N.nan, N.nan
    if len(x) < 2:
        return x[0], N.nan, N.nan

    sum_w = N.sum(w)
    sum_w2 = N.sum(w*w)

    xm = N.dot(w,x)/sum_w

    # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights
    var = sum_w/(sum_w**2 - sum_w2) * N.sum(w * (x - xm)**2) # unbiased sample variance -
    N_eff = sum_w**2/sum_w2
    var_avg = var/N_eff # variance of the mean - scaled by the effective sample size Neff = sum_w**2/sum_w2

    return xm, N.sqrt(var), N.sqrt(var_avg)


def w_avg_std_array(x, w):
    """
    Returns weighted avg and unbiased std dev for multiple samples on 0. axis

    x - 2 dim ndarray 
        axis 0 -> do not care (e.g. samples in time, energy...)
        axis 1 -> samples to be avg. and std

    w - 1 dim ndarray
        weights of axis1 in z

    Returns two 1 dim ndarrays:
    avg - weighted averages   len == len(x[0])
    std - unbiased std. dev.  len == len(avg)
    """
    z = x.T
    xlen = len(x[0])
    zlen = len(z[0])

    if zlen != len(w):
        print("AAAAAAAAAAAAAA!!!!!")

    avg = N.empty(xlen)
    std = N.empty(xlen)
    std_avg = N.empty(xlen)
    for i in range(xlen):
        avg[i], std[i], std_avg[i] = w_avg_std(z[i], w)
    
    return avg, std, std_avg

def w_avg_std_param(param, samples, w, rtol=1.0000000000000001e-05, atol=1e-08):
    """
    Returns weighted avg and unbiased std dev for multiple samples as a
    function of param. Assumes that the probability distribution
    of each sample is a function of the corresponding param. Therefore,
    only samples with almost equal params are averaged. Params a and b
    are assumed equal if
        abs(a - b) <= (atol + rtol * abs(b))

    param - 1 dim ndarray 
        parametrization of the samples

    samples - 1 dim ndarray
        samples to be avg. and std

    w - 1 dim ndarray
        weights of the samples

    Returns three 1 dim ndarrays:
    param - weighted averages   len == len(x[0])
    avg - weighted averages   len == len(x[0])
    std - unbiased std. dev.  len == len(avg)
    """

    if len(param) != len(samples) or len(param) != len(w):
        raise RuntimeError("w_avg_std_unsorted: input array lengths don't match")

    # first sort everything according to param
    sort_indices = N.argsort(param)
    param = param[sort_indices]
    samples = samples[sort_indices]
    w = w[sort_indices]

    # now find the unique values
    is_different = N.zeros_like(param, dtype=bool)
    is_different[0] = True
    is_different[1:] = N.abs(N.diff(param)) > (atol+rtol*N.abs(param[1:]))
    diff_indices = N.arange(len(param))[is_different]

    xlen = len(diff_indices)
    avg = N.empty(xlen)
    std = N.empty(xlen)
    std_avg = N.empty(xlen)

    for i in range(xlen):
        i1 = diff_indices[i]
        if i < xlen-1:
            i2 = diff_indices[i+1]
        else:
            i2 = len(param)
        avg[i], std[i], std_avg[i] = w_avg_std(samples[i1:i2], w[i1:i2])
    
    return param[diff_indices], avg, std, std_avg


if __name__ == "__main__":
    """
    very simple test for w_avg_std()
    """

    x = N.array([1.,2.,3.,4.,5.])
    w = N.array([1.,1.,1.,1.,1.])

    print("x =", x)
    print("w =", w)
    print("N.average(x,w) =", N.average(x,weights=w), "\t" "N.std(x,ddof=1) =", N.std(x,ddof=1))

    print("w_avg_std(x, w) =", w_avg_std(x, w))
