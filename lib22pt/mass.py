import numpy as np

class Mass:
    def __init__(self, fname):
        self.fname = fname
        import re
        fr = open(fname)

        for lineno, line in enumerate(fr):
            if re.search("// Integration \(s\)*", line):
                self.integration =\
                    float(re.search("// Integration \(s\) ([0-9.]+)*", line).group(1))
            if re.search("// Iterations*", line):
                self.iterations =\
                    float(re.search("// Iterations ([0-9]+)*", line).group(1))
            if re.search("// 8P\. *", line):
                self.pot_8P =\
                    float(line.split()[2])
            if re.search("Mass*", line):
                firstline = lineno
        fr.close()

        try:
            self.data = np.loadtxt(fname, skiprows=firstline+2)[1:-1, [0, 1, 6]]
        except StopIteration as e:
            print("Mass: no data in file %s" % fname)
            raise
        self.data[:,[1,2]] *= self.iterations
        #self.data[:,[1,2]] /= self.integration
        #self.data[:,0] += self.pot_8P
    def merge(self, scan):
        self.data = np.vstack((self.data, scan.data))

    def plot(self, ax=None, show=False,
            plotargs=dict(linestyle="-", color="C0", linewidth=2), offset=0.5, significant_only=True, decorate=True):
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter

        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()

        if significant_only:
            ax.plot(self.data[:,0]-offset, self.data[:,1]-self.data[:,2], label=self.fname,\
                **plotargs)
        else:
            ax.errorbar(self.data[:,0]-offset, self.data[:,1], yerr=self.data[:,2], label=self.fname,\
                **plotargs)

        if decorate:
            ax.set_yscale("symlog", linthreshy=1, linscaley=0.5)
            ax.set_xlabel("mass (amu)")
            ax.set_ylabel("counts")
            ax.set_ylim(ymin=0)
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%d"))

            ax.grid(linewidth=1)
            ax.grid(which="minor", color="0.5")
            ax.legend(fontsize=8)

        if show == True:
            plt.show()

        return ax

