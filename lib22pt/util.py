import numpy as np
import os


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
            "cm3mol":first virial coefficient https://doi.org/10.1063/1.1840728
        
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
    elif aunit == "cm3mol":
        alpha_A3 = alpha/(4/3*np.pi*sc.N_A)*1e24
        b0 = sc.physical_constants["Bohr radius"][0]
        alpha_SI = alpha_A3 * 1e-30 * 4*sc.pi*sc.epsilon_0
    else:
        raise(ValueError("aunit " + repr(aunit) + " not understood"))


    # calculate the rate coefficient in SI units
    k_SI = sc.e/(2*sc.epsilon_0)*np.sqrt(alpha_SI/mu_SI)

    return 1e6*k_SI


def k_ADO(T, alpha, muD, m1, m2, aunit="SI", munit="SI", method="Su_Chesnavich"):
    """Calculate capture rate coefficient of ion-polar-molecule collisions
    within the Average Dipole Orientation (ADO) theory. A good source of the neutral
    data is the NIST CCCBDB https://cccbdb.nist.gov/exp1x.asp

    Args:
        T: temperature, can be numpy array
        alpha: neutral polarizability (in SI (C2 m2 J-1) by default)
                for H2O, alpha = 9.93 # au https://doi.org/10.1103/PhysRevLett.94.013204
                for NH3, alpha = 14.19 # au https://doi.org/10.1063/1.1673524
        muD: neutral dipole moment (in SI (C m) by default)
                for H2O, muD = 0.73 # au https://doi.org/10.1063/1.460471
                for NH3, muD = 1.476 # Debye https://doi.org/10.1016/S0301-0104(97)00145-6
        m1, m2: reactant masses [amu]
        aunit: unit of polarizability, can be:
            "A3": Angstrom^3
            "SI": C m^2 V^-1 (default)
            "au": atomic units, (Bohr radius)^3
        munit: unit of dipole moment, can be:
            "Debye": 1e-10 statC Angstrom
            "SI": C m (default)
            "au": atomic units, (Bohr radius * q0)^3
        method: method of calculation. The implemented methods are:
            "Su_Chesnavich": https://doi.org/10.1063/1.442828 (default)
            "Bates": https://doi.org/10.1016/0009-2614(81)85405-X


    Returns:
        array of ADO rate coefficients [cm^3 s^-1]
    """
    from lib22pt.util import langevin
    import scipy.constants as sc

    # convert alpha to SI units
    if aunit == "A3":
        alpha_SI = alpha * 1e-30 * 4*sc.pi*sc.epsilon_0
    elif aunit == "SI":
        alpha_SI = alpha
    elif aunit == "au":
        # https://physics.nist.gov/cgi-bin/cuu/Value?auepol|search_for=electronmass
        b0 = sc.physical_constants["Bohr radius"][0]
        alpha_SI = alpha * b0**3 * 4*sc.pi*sc.epsilon_0
    else:
        raise(ValueError("aunit " + repr(aunit) + " not understood"))

    # convert muD to SI units
    if munit == "Debye":
        muD_SI = 1e-21/sc.c*muD
    elif munit == "SI":
        muD_SI = muD
    elif munit == "au":
        # https://physics.nist.gov/cgi-bin/cuu/Value?auedm
        b0 = sc.physical_constants["Bohr radius"][0]
        muD_SI = b0**3 * sc.e * muD
    else:
        raise(ValueError("munit " + repr(munit) + " not understood"))

    k_Lang = langevin(alpha=alpha_SI, m1=m1, m2=m2, aunit="SI")

    T = np.atleast_1d(T)
    if method=="Su_Chesnavich":
        # (Su and Chesnavich 1982) https://doi.org/10.1063/1.442828
        x = muD_SI/np.sqrt(2*alpha_SI*sc.k*T)
        res = 0.4767*x + 0.6200
        res[x < 2] = (x[x<2] + 0.5090)**2/10.526 + 0.9754
    elif method=="Bates":
        # alternate formulation by (Bates 1981) https://doi.org/10.1016/0009-2614(81)85405-X
        beta = 2*alpha_SI*sc.k*T/muD_SI**2
        res = 1.105/np.sqrt(np.pi*beta)
    else:
        raise(ValueError("method " + repr(method) + " not understood"))
    return res*k_Lang


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


def decimate_dataframe(dataframe, bins, refcol="T22PT", errtype="sample_weights", dropnan=True, strcols=["note", "Viscovac"], add_errors=True):
    import pandas as pd
    """decimate pandas dataframe by binning values in refcol into bins.
    See avg.weighted_mean for explanation of the errtype.
    strcols defines the names of columns that should be interpreted as strings, i.e., not averaged
    if add_errors: error estimates are added to numerical columns without error estimates"""

    from lib22pt.avg import w_avg_std, wstd_avg_std, weighted_mean
    cols_w_errs = []
    cols = list(dataframe.columns)
    for col in dataframe.columns:
        if col+"_err" in cols:
            cols_w_errs.append(col)
            cols.remove(col)
            cols.remove(col+"_err")
    cols_wo_errs = cols

    allcols = cols_wo_errs + cols_w_errs
    averages = pd.DataFrame(index=range(len(bins)-1), columns=dataframe.columns)
    for i in range(len(bins)-1):
        indices = (dataframe[refcol] >= bins[i] ) & (dataframe[refcol] < bins[i+1])
        if not np.any(indices): continue
        subset = dataframe.loc[indices]

        for col in cols_w_errs:
            if len(subset[col]) > 1:
                averages.loc[i,col], averages.loc[i,col+"_err"], *_ =\
                        weighted_mean(subset[col].values, std=subset[col+"_err"].values, dropnan=True, errtype=errtype)

            else:
                averages.loc[i,col], averages.loc[i,col+"_err"] =\
                        subset[col].iloc[0], subset[col+"_err"].iloc[0]

        for col in cols_wo_errs:
            if isinstance(subset[col].iloc[0], (tuple)):
                averages.loc[i, col] = str(sum(list(subset[col]), ()))
            elif isinstance(subset[col].iloc[0], (str)) or col in strcols:
                averages.loc[i, col] = "".join(map(str, subset[col]))
            elif len(subset[col].dropna()) > 0:
                averages.loc[i,col] = np.nanmean(subset[col].astype(float))
                if add_errors:
                    averages.loc[i,col+"_err"] = np.nanstd(subset[col].astype(float))
    return averages


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


def shift_T(df, Tcol, ncols, kcols, T_shift, T_e_low, T_e_high, sys_only=False):
    import pandas as pd
    df[Tcol+"_shift"] = df[Tcol] + T_shift
    f = np.sqrt(df[Tcol+"_shift"]/df[Tcol])

    for ncol in ncols:
        try:
            df[ncol+"_shift"] = df[ncol]/f
        except(KeyError):
            pass # we are probably dealing with metadata dict

    for kcol in kcols:
        df[kcol+"_shift"] = df[kcol]*f
        df[kcol+"_shift_err"] = df[kcol+"_err"]*f

    # temperature error is nonlinear and non-random. Determine the total error bound by interval arithmetic
    f_low = np.sqrt((df[Tcol+"_shift"]-T_e_low)/df[Tcol+"_shift"])
    f_high = np.sqrt((df[Tcol+"_shift"]+T_e_high)/df[Tcol+"_shift"])
    for kcol in kcols:
        if sys_only:
            # these errors are purely systematic
            df[kcol+"_shift_loerr"] = df[kcol+"_shift"]*(1-f_low)# + df[kcol+"_err"]*f_low
            df[kcol+"_shift_hierr"] = df[kcol+"_shift"]*(f_high-1)# + df[kcol+"_err"]*f_high
        else:
            # include stat errors
            df[kcol+"_shift_loerr"] = df[kcol+"_shift"]*(1-f_low) + df[kcol+"_err"]*f_low
            df[kcol+"_shift_hierr"] = df[kcol+"_shift"]*(f_high-1) + df[kcol+"_err"]*f_high

    # sum temperature errors if available
    if isinstance(df, pd.DataFrame) and Tcol+"_err" in df.columns:
        T_err = df[Tcol+"_err"]
    else:
        T_err = 0.
    df[Tcol+"_shift_loerr"] = T_err + T_e_low
    df[Tcol+"_shift_hierr"] = T_err + T_e_high

    return


def plot_avg_shifted(data_avg, name, ax, xfun=lambda x:x, plotargs=dict(label="", color="k", fmt="o", zorder=2), errorbar=True, err="err"):
    if errorbar:

        if err=="avgerr":
            print("ERROR = ")
            print(data_avg[name+"_shift_lo"+err])
        ax.errorbar(xfun(data_avg["T22PT_shift"]), data_avg[name+"_shift"],
            yerr=np.vstack((
                data_avg[name+"_shift_lo"+err],
                data_avg[name+"_shift_hi"+err]
            )),
            xerr=np.vstack((
                -xfun(data_avg["T22PT_shift"]-data_avg["T22PT_shift_loerr"]) + xfun(data_avg["T22PT_shift"]),
                 xfun(data_avg["T22PT_shift"]+data_avg["T22PT_shift_hierr"]) - xfun(data_avg["T22PT_shift"])
                 )),
            **plotargs)
    else:
        #fmt = plotargs.pop("fmt", "o")
        ax.errorbar(xfun(data_avg["T22PT_shift"]), data_avg[name+"_shift"], **plotargs)


def plot_shifted(data, name, ax, xfun=lambda x:x, plotargs=dict(label="", color="k", fmt="o", zorder=2)):
    ax.errorbar(xfun(data["T22PT_shift"]), data[name+"_shift"],
            yerr=data[name+"_shift_err"],
            **plotargs)


def set_unit(axis, unit):
    """Set units of a matplotlib axis. Usage: set_unit(ax1.yaxis, 1e-9) """
    from matplotlib.ticker import FuncFormatter
    axis.set_major_formatter(
        FuncFormatter(
            lambda x, pos: np.round(x/unit, decimals=3)
        )
    )


def print_banner(text, ch='#', length=78):
    spaced_text = ' %s ' % text
    banner = spaced_text.center(length, ch)
    print("\n" + ch*length + "\n" + banner + "\n" + ch*length + "\n")



def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def dict2Params(dic):
    from lmfit import Parameters
    if isinstance(dic, Parameters): return dic.copy()
    p = Parameters()
    for key, val in dic.items():
        p.add(key, value=val)
    return p
