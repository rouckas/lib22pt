import numpy as np
import pandas as pd
from os.path import splitext, basename

from .util import dict2Params, concentration, ensure_dir, warn, decimate_dataframe
from .rate import MultiRate

def Tconverter(cell):
    if isinstance(cell, (float, int)): return cell
    elif cell.strip() == "": return np.nan
    toks = cell.split(" - ")
    try:
        if len(toks) == 1:
            res = float(toks[0])
        else:
            res = (float(toks[0])+float(toks[1]))/2
    except ValueError as e:
        raise ValueError("Cannot read temperature from the excel file: string '%s'"%str(cell))
    return res
    
def rateconverter_lower(cell):
    return cell.lower().replace(u"\xa0", "") + ".dat"

def rateconverter(cell):
    return cell.replace(u"\xa0", "") + ".dat"
    
def strconverter(cell):
    return cell.replace(u"\xa0", "")
    
converters_lower = {"T22PT":Tconverter, "note":strconverter, "Viscovac":strconverter, "Plaser_mW":Tconverter, "rate":rateconverter_lower}
converters = {"T22PT":Tconverter, "note":strconverter, "Viscovac":strconverter, "Plaser_mW":Tconverter, "rate":rateconverter}


def plot_title(row, mdata, config):
    T_shift = mdata["T_shift"]
    Tcol = "T22PT"

    title = "$T_{\\rm 22PT} + %.0f\\ {\\rm K} = %.1f\\,\\rm K$   " % (T_shift, row["T22PT"]+T_shift) 
    for key, val in row.items():
        if key[0] == "p":
            f_key = "f_" + key[1:]
            if f_key in mdata:
                n = concentration(row[Tcol]+T_shift, row[key], mdata[f_key], 0, 0)
                title += "[%s] = %.2g $\\rm cm^{-3}$ " % (key[1:], n)

    return title

def fit_dataset(data, mdata, config, ID, fit_methods, plot_title=plot_title):
    from data import p0 as _p0
    plot = config.get("plot_all", mdata["replot"])
    refit = config.get("refit_all", mdata["refit"])


    if refit:
        fitrange = mdata.get("fitrange", (0, None))
        fitmask = np.s_[fitrange[0]:fitrange[1]]
        method = mdata["method"]

        ensure_dir(config["paramdir"])
        ensure_dir(config["ratefitdir"] + "/" + ID + "/")
        if plot:
            rateplotdir = config["rateplotdir"]+"/"+ID+"/"
            ensure_dir(rateplotdir)


        for i, d in data.iterrows():
            if data.loc[i, "good"] == "SKIP":
                continue
            
            print(d["rate"])
            p0 = mdata.get("p0", _p0[method])
            R = MultiRate(config["ratedir"] + "/" + d["rate"], t_offset=config.get("t_offset", 0.))

            if data.loc[i, "good"] != "NOFIT":
                fit_method = fit_methods.get(method, None)
                if fit_method is None:
                    raise(RuntimeError("Fit method name '%s' not known" % mdata["method"]))

                new_data = fit_method(R , p0, fitmask, mdata["columns"])
                data.loc[i, new_data.keys()] = new_data.values()

            if mdata.get("reuse_p", False) and R.fitresult is not None:
                # print("saving p0")
                p0 = R.fitresult.params
            if plot:
                plotname = splitext(basename(d["rate"]))[0] + config["plotext"]
                title = plot_title(data.loc[i], mdata, config)
                R.plot_to_file(rateplotdir+plotname, comment=title, plot_columns=mdata.get("plot_columns", mdata["columns"]))
            data.to_excel(config["paramdir"] + "/" + ID + ".xlsx", sheet_name="Sheet1")
            R.save_data_fit_excel(config["ratefitdir"] + "/" + ID + "/" + splitext(basename(d["rate"]))[0] + ".xlsx",
                metadata={"params": data.loc[i], "metadata": pd.Series(mdata), "config": pd.Series(config),
                          "fitparams":pd.Series(R.fitresult.params.valuesdict() if R.fitresult is not None else {}, dtype=np.float64)})
          
          
    else:
        fitresults = pd.read_excel(config["paramdir"] + "/" + ID + ".xlsx", "Sheet1")
        # If you want to index by the rate name later, you may set the index in data.py as:
        #   datasets[ID]["_rate"] = datasets[ID]["rate"]
        #   datasets[ID].set_index("_rate", inplace=True)
        # and then read the data here as
        # fitresults = pd.read_excel(config["paramdir"] + "/" + ID + ".xlsx", "Sheet1", index_col="_rate")

        for c in fitresults.columns:
            if c not in data.columns:
                data[c] = fitresults[c]
            

def subtract_BG(data, mdata, config, ID, rcols):
    """ simple constant background rate subtraction. """
    if "good" in data.columns:
        good = data["good"] == "GOOD"
        bg = data["good"] == "BG"
    else:
        good = data.index

    from lib22pt.avg import weighted_mean
    for rcol in rcols:
        if np.count_nonzero(np.isfinite(data[rcol][bg])) == 0:
            warn("BG not found in dataset " + ID + " assuming zero BG", 1)
        else:
            if len(data[rcol][bg]) > 1:
                r_bg, r_bg_err = weighted_mean(data[rcol][bg], std=data[rcol+"_err"][bg], dropnan=True, errtype="sample_weights")
            else:
                r_bg = data[rcol][bg].iloc[0]
                r_bg_err = data[rcol+"_err"][bg].iloc[0]
            data[rcol] -= r_bg
            data[rcol+"_err"] = np.sqrt(data[rcol+"_err"]**2 + r_bg_err**2)


def calculate_n(data, mdata, config, ID, pcols, ncols, Tcol="T22PT"):
    for pcol, ncol in zip(pcols, ncols):
        fcol = "f_"+pcol[1:]
        data[ncol] = concentration(data[Tcol], data[pcol], mdata[fcol], 0, 0)
   

def calculate_k(data, mdata, config, ID, rcols, kcols, ncol, Tcol="T22PT", k3cols=None):
    if "good" in data.columns:
        good = data["good"] == "GOOD"
    else:
        good = data.index

    for col in rcols:
        if col not in data.columns:
            data[col] = np.nan
            data[col+"_err"] = np.nan

    n = data[ncol]
    for rcol, kcol in zip(rcols, kcols):
        data.loc[good, kcol] = data.loc[good, rcol]/n
        data.loc[good, kcol+"_err"] = data.loc[good, rcol+"_err"]/n

    if k3cols is not None:
        for rcol, kcol in zip(rcols, k3cols):
            data.loc[good, kcol] = data.loc[good, rcol]/n**2
            data.loc[good, kcol+"_err"] = data.loc[good, rcol+"_err"]/n**2

def average_datasets(datasets, datasets_to_avg, bins):
    if len(datasets_to_avg):
        averaged = True
        data_k_to_avg = pd.concat([datasets[ID] for ID in datasets_to_avg])
        k_avg = decimate_dataframe(data_k_to_avg, bins)
    else:
        averaged = False
        k_avg = None
        data_k_to_avg = None
    return averaged, k_avg, data_k_to_avg

def fit_ndeps(datasets, metadata, config, ndeps, rcols, kcols, ncol="nH2_shift"):
    def f(x, a, b): return a*x + b
    from scipy.optimize import curve_fit

    ndep_coeffs = []
    for ID in ndeps:
        dataset = datasets[ID]
        Tmean = np.mean(dataset["T22PT_shift"])
        Tstd = np.std(dataset["T22PT_shift"])
        metadata[ID]["T22PT_shift"] = Tmean
        line = {"ID": ID, "T22PT_shift": Tmean, "T22PT_shift_err": Tstd}
        for kname, ratename in zip(kcols, rcols):
            kname = kname + "_shift"
            n_err = metadata[ID]["n_sys_err"]
            good = (dataset[ratename] > dataset[ratename+"_err"]) & (dataset["good"] == "GOOD")
            popt, pcov = curve_fit(f, dataset[ncol][good], dataset[ratename][good],
                sigma = dataset[ratename+"_err"][good] + dataset[ratename][good]*n_err)
            perr = np.sqrt(np.diag(pcov))
            line[kname] = popt[0]
            line[kname+"_err"] = perr[0]
            line[ratename+"_bg"] = popt[1]
            line[ratename+"_bg_err"] = perr[1]

        metadata[ID].update(line)
        ndep_coeffs.append(line)
    return pd.DataFrame(ndep_coeffs)
