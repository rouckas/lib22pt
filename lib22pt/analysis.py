import numpy as np
import pandas as pd
from os.path import splitext, basename

from .util import dict2Params, concentration, ensure_dir, warn
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
    T_shift = config["T_shift"]
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
        p0 = dict2Params(mdata.get("p0"))
        fitmask = np.s_[fitrange[0]:fitrange[1]]
        columns = mdata["columns"]
        method = mdata["method"]

        ensure_dir(config["fitdir"])
        ensure_dir(config["ratefitdir"] + "/" + ID + "/")
        if plot:
            rateplotdir = config["rateplotdir"]+"/"+ID+"/"
            ensure_dir(rateplotdir)


        for i, d in data.iterrows():
            if data.loc[i, "good"] == "SKIP":
                continue
            
            print(d["rate"])
            p0 = _p0[method]
            R = MultiRate(config["ratedir"] + "/" + d["rate"], t_offset=config.get("t_offset", 0.))

            if data.loc[i, "good"] != "NOFIT":
                fit_method = fit_methods.get(method, None)
                if fit_method is None:
                    raise(RuntimeError("Fit method name '%s' not known" % mdata["method"]))

                new_data = fit_method(R , p0, fitmask, mdata["columns"])
                data.loc[i, new_data.keys()] = new_data.values()

            if mdata.get("reuse_p", False) and R.fitresult is not None:
                print("saving p0")
                p0 = R.fitresult.params
            if plot:
                plotname = splitext(basename(d["rate"]))[0] + config["plotext"]
                title = plot_title(data.loc[i], mdata, config)
                R.plot_to_file(rateplotdir+plotname, comment=title, plot_columns=mdata.get("plot_columns", mdata["columns"]))
            data.to_excel(config["fitdir"] + "/" + ID + ".xlsx", sheet_name="Sheet1")
            R.save_data_fit_excel(config["ratefitdir"] + "/" + ID + "/" + splitext(basename(d["rate"]))[0] + ".xlsx",
                metadata={"params": data.loc[i], "metadata": pd.Series(mdata), "config": pd.Series(config),
                          "fitparams":pd.Series(R.fitresult.params.valuesdict() if R.fitresult is not None else {}, dtype=np.float64)})
          
          
    else:
        fitresults = pd.read_excel(config["fitdir"] + "/" + ID + ".xlsx", "Sheet1", index_col="_rate")
        for c in fitresults.columns:
            if c not in data.columns:
                data[c] = fitresults[c]
            

def subtract_BG(data, mdata, config, ID, rcols):
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
            r_bg, r_bg_err = weighted_mean(data[rcol][bg], std=data[rcol+"_err"][bg], dropnan=True, errtype="sample_weights")
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