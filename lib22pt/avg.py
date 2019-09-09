#!/usr/bin/python
import numpy as N

def wstd_avg_std(x, std, axis=0):
    """
    Returns weighted mean and standard deviation of samples with known
    standard deviations. Optimal weighting is used.
    x - samples (ndarray)
    w - standard deviations of samples (ndarray)
    """

    if N.shape(x)[axis] < 2:
        return N.rollaxis(x, axis)[0], N.rollaxis(std, axis)[0]


    if len(x) < 2:
        return x[0], std[0]

    w = 1/std**2
    sum_w = N.sum(w, axis=axis)

    xm = N.sum(w*x, axis=axis)/sum_w
    var = 1/sum_w

    return xm, N.sqrt(var)

def w_avg_std(x, w, dropnan=False):
    """
    Returns weighted mean and sqrt(weighted unbiased sample variance),
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

    var = sum_w/(sum_w**2 - sum_w2) * N.sum(w * (x - xm)**2)
    var_avg = var*sum_w2/sum_w**2

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
