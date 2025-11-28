import numpy as np
import matplotlib.pyplot as plt
from lib22pt.util import ensure_dir
from os.path import join


def plot_setup(fontsize=10, figsize=(3.2, 2.5)):
    plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
    plt.rcParams['axes.xmargin'] = 0

    plt.rcParams["xtick.major.size"] = 5
    plt.rcParams["xtick.minor.size"] = 3
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.major.size"] = 5
    plt.rcParams["ytick.minor.size"] = 3
    plt.rcParams["ytick.direction"] = "in"

    plt.rcParams["xtick.top"] = True
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True
    plt.rcParams["ytick.right"] = True

    plt.rcParams["axes.labelsize"] = fontsize
    plt.rcParams["axes.titlesize"] = fontsize
    plt.rcParams["xtick.labelsize"] = fontsize
    plt.rcParams["ytick.labelsize"] = fontsize

    plt.rcParams["figure.titlesize"] = fontsize
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["legend.framealpha"] = 1

    plt.rcParams['figure.figsize'] = figsize   # figure size in inches
    plt.rcParams['grid.linewidth'] = 0.5
    ensure_dir("pdf")
    ensure_dir("png")

def savefig_pdf_png(fig, basename, directory="."):
    fig.savefig(join(directory, "pdf", basename+".pdf"))
    fig.savefig(join(directory, "png", basename+".pdf"), dpi=300)

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

def plot_k_T(ax, datasets, metadata, config, to_plot, kcol, Tcol="T22PT_shift", title_reaction="", yscale="linear", xunit=1, yunit=1e-9,
             plotargs={}):

    if ax is None:
        f = plt.figure(figsize=(7,5))
        ax = f.add_axes([.17, .12, .79, .8])
    else:
        f = ax.get_figure()

    for ID in to_plot:
        data = datasets[ID].query("good == 'GOOD'")
        color = metadata[ID]["color"]
        label=ID
        mfc=color
        fmt="o"
        print("plotting", ID)
        _plotargs = dict(color=color, fmt=fmt, mfc=mfc, label=label, zorder=2)
        _plotargs.update(metadata[ID].get("plotargs", {}))
        _plotargs.update(plotargs)
        
        ax.errorbar(data[Tcol]/xunit, data[kcol]/yunit, yerr=data[kcol+"_err"]/yunit,
                     **_plotargs)

    xunitlabel = "10^{%d}"%np.log10(xunit) if xunit != 1 else ""
    yunitlabel = "10^{%d}"%np.log10(yunit) if yunit != 1 else ""
    ax.set_ylabel(r"$k$ ($\rm %s cm^6{s}^{-1}$)"%yunitlabel)
    ax.set_xlabel(r"$T (\rm %s K)$"%xunitlabel)
    ax.legend(fontsize=7)
    ax.set_yscale(yscale)
    ax.grid()
    ax.set_title(title_reaction + " rate coefficient")

    return f, ax
    
def plot_coeff(ax, datasets, metadata, config, to_plot, ycol, xcol="T22PT_shift",
                title="", yscale="linear", xunit=1, yunit=1,
                  xlabel=r"$T (\rm {xunitlabel} K)$",
                  ylabel=r"$k (\rm {yunitlabel} cm^3 {{s}}^{{-1}})$",
                  plotargs = {}):
    #f = plt.figure(figsize=(7,5))
    #ax = f.add_axes([.17, .12, .79, .8])
    for ID in to_plot:
        data = datasets[ID].query("good == 'GOOD'")
        color = metadata[ID]["color"]
        label=ID
        mfc=color
        fmt="o"
        print("plotting", ID)
        _plotargs = dict(color=color, fmt=fmt, mfc=mfc, label=label, zorder=2)
        _plotargs.update(metadata[ID].get("plotargs", {}))
        _plotargs.update(plotargs)
        ax.errorbar(data[xcol]/xunit, data[ycol]/yunit, yerr=data[ycol+"_err"]/yunit,
                    **_plotargs)

    xunitlabel = "10^{%d}"%np.log10(xunit) if xunit != 1 else ""
    yunitlabel = "10^{%d}"%np.log10(yunit) if yunit != 1 else ""
    ax.set_ylabel(ylabel.format(**locals()))
    ax.set_xlabel(xlabel.format(**locals()))
    ax.legend(fontsize=7)
    ax.set_yscale(yscale)
    ax.grid(True)
    ax.set_title(title)

def plot_ndeps(datasets, metadata, config, ndeps, rcols, kcols, ncol="nH2_shift",
        xunit = 1e10, kunit = 1e-9):
    def f(x, a, b): return a*x + b
    for ID in ndeps:
        ensure_dir(config["rateplotdir"] + "/ndeps/")

        fig, ax = plt.subplots(1,1)
        dataset = datasets[ID]
        for kname, ratename in zip(kcols, rcols):
            kname = kname + "_shift"
            n_err = metadata[ID]["n_sys_err"]
            good = (dataset[ratename] > dataset[ratename+"_err"]) & (dataset["good"] == "GOOD")
            ax.errorbar(dataset[ncol][good]/xunit, dataset[ratename][good],
                        xerr=dataset[ncol][good]*n_err/xunit,
                        yerr=dataset[ratename+"_err"][good], fmt="o")
            nrange = np.linspace(dataset[ncol][good].min(), dataset[ncol][good].max())
            ax.plot(nrange/xunit, f(nrange, metadata[ID][kname], metadata[ID][ratename+"_bg"]))
            ax.set_xlabel(ncol + "($10^{%d} \\rm cm^{-3}$)"%np.log10(xunit))
            ax.set_ylabel("$r\\ (\\rm s^{-1})$")
            ax.text(0.05, 0.9, "$k = (%.2f\\pm%.2f)\\times\\ 10^{%d}\\ \\rm cm^3 s^{-1}$"%
                    (metadata[ID][kname]/kunit, metadata[ID][kname+"_err"]/kunit, int(np.log10(kunit))),
                    transform=ax.transAxes)
            ax.text(0.05, 0.85, "$r_{bg} = (%.2f\\pm%.2f)\\ s^{-1}$"%
                    (metadata[ID][ratename+"_bg"], metadata[ID][ratename+"_bg_err"]),
                    transform=ax.transAxes)
            ax.set_title("ID: " + ID + ", $ T = (%.1f \\pm %.1f^{\\rm stat} \\pm 5^{\\rm sys})\\ \\rm K$" %
                        (metadata[ID]["T22PT_shift"], metadata[ID]["T22PT_shift_err"]))
        fig.savefig(config["rateplotdir"] + "/ndeps/" + ID + "_ndep.pdf")
