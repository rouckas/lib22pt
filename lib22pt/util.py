import numpy as np
import os


def concentration(temp, Paml22PT, f_22PT, PamlSIS=0., f_SIS=0., amltemp=300):
    """Calculate gas number density in the trap [cm^-3] from outside pressure in Pa
    
    Args:
        temp: trap temperature in K
        Paml22PT: pressure on the outside gauge due to gas added into the trap
        f_22PT: calibration factor for adding gas into the trap
            typical values are 45 for H2 and 140 for He
        PamlSIS: pressure on the outside gauge added from outside (e.g., SIS)
        f_SIS: calibration factor for adding gas into the trap
            typical value is 1.5 for H2 and 5 for He
        amltemp: temperature of the outer gauge

    Returns:
        number density in cm^-3

    Works under the assumption of pure molecular regime during measurement
    and calibration
    """

    from scipy.constants import k as k_B

    return (Paml22PT*f_22PT + PamlSIS*f_SIS)/k_B/np.sqrt(amltemp*temp)/10000



def langevin(alpha, m1, m2, aunit="A3"):
    """Calculate Langevin reaction rate coefficient of ion-molecule reaction
    
    Args:
        alpha: neutral polarizability (in Angstrom^3 by default)
                for H2, alpha = 0.786 A^3
        m1, m2: reactant masses [amu]
        aunit: unit of polarizability, can be:
            "A3": Angstrom^3 (default)
            "SI": C m^2 V^-1
            "au": atomic units, (Bohr radius)^3
        
    Returns:
        langevin reaction rate coefficient [cm^3 s^-1]
    """

    # load physical constants
    from scipy import constants as sc
    amu = sc.physical_constants["atomic mass constant"][0]

    # calculate reduced mass in SI units
    mu_SI = m1*m2/float(m1+m2)*amu

    # convert alpha to SI units
    if aunit == "A3":
        alpha_SI = alpha * 1e-30 * 4*sc.pi*sc.epsilon_0
    elif aunit == "SI":
        alpha_SI = alpha
    elif aunit == "au":
        b0 = sc.physical_constants["Bohr radius"][0]
        alpha_SI = alpha * b0**3 * 4*sc.pi*sc.epsilon_0

    # calculate the rate coefficient in SI units
    k_SI = sc.e/(2*sc.epsilon_0)*np.sqrt(alpha_SI/mu_SI)

    return 1e6*k_SI



def decimate(data, bins, add_errs=False):
    """ the data columns are [data_x, data_y, y_err] """

    from .avg import w_avg_std, wstd_avg_std
    averages = np.zeros((len(bins)-1, 5))*np.nan
    for i in range(len(bins)-1):
        indices = (data[:,0] >= bins[i] ) & (data[:,0] < bins[i+1])
        if np.any(indices): averages[i,4] = 1
        else: continue
        subset = data[indices]

        averages[i,0], averages[i,2], dum = w_avg_std(subset[:,0], 1/subset[:,2]**2)
        averages[i,1], averages[i,3], dum = w_avg_std(subset[:,1], 1/subset[:,2]**2)
        if add_errs:
            # for example to account for stat spread of data + fit quality
            averages[i,3] += wstd_avg_std(subset[:,1], subset[:,2])[1]
    return averages



def polysmooth(points, xdata, ydata, wlen, deg, leastdeg=None, deriv=0, logwindow=False,\
        kernel="step", nan=np.nan):
    points = np.atleast_1d(points)
    res = np.zeros_like(points)
    if leastdeg is None: leastdeg = deriv
    elif min(leastdeg, deg) < deriv:
        raise RuntimeError("polysmooth: poly degree must be >= deriv")
    for i, point in enumerate(points):
        if kernel == "step":
            if logwindow:
                wmin, wmax = point/wlen, point*wlen
            else:
                wmin, wmax = point-wlen, point+wlen
            I = (xdata>wmin) & (xdata<wmax)
            nI = np.count_nonzero(I)

            degnow = min([nI-1, deg])
            if degnow < leastdeg:
                res[i] = nan
                continue
            x, y = xdata[I], ydata[I]
            weights = None

        elif kernel=="gauss":
            gaussian = lambda x, mu, sigma:\
                np.exp(-(x-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))

            if logwindow:
                weights = gaussian(np.log10(xdata), np.log10(point), np.log10(wlen))
            else:
                weights = gaussian(xdata, point, wlen)

            x, y = xdata, ydata

        try:
            p = np.polyfit(x, y, deg, w=weights)
        except np.lib.polynomial.RankWarning:
            res[i] = np.nan
        else:
            res[i] = np.polyder(np.poly1d(p), m=deriv)(point)
    return res



def stitch(avg1, avg2, debug=False):
    """Find multiplier for avg1 to match avg2 in the overlapping region.
    Both arrays must have the same x-axis."""
    def distance(p, data_avg, averages):
        overlap = (avg1[:,4]>0) & (avg2[:,4]>0)
        dist = avg1[overlap,1]*p[0] - avg2[overlap,1]
        var = np.sqrt((avg1[overlap,3]*p[0])**2 + avg2[overlap,3]**2)
        if debug: print(p, overlap.astype(int), dist)
        return (dist/var)[~np.isnan(dist/var)]

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



def print_banner(text, ch='#', length=78):
    spaced_text = ' %s ' % text
    banner = spaced_text.center(length, ch)
    print("\n" + ch*length + "\n" + banner + "\n" + ch*length + "\n")



def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)
