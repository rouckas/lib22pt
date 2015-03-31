import numpy as np

class Mass:
    def __init__(self, fname):
        self.fname = fname
        self.data = np.loadtxt(fname, skiprows=8)[1:-1, [0, 1, 6]]

        import re
        fr = open(fname)

        for line in fr:
            if re.search("// Integration \(s\)*", line):
                self.integration =\
                    float(re.search("// Integration \(s\) ([0-9.]+)*", line).group(1))
            if re.search("// Iterations*", line):
                self.iterations =\
                    float(re.search("// Iterations ([0-9]+)*", line).group(1))
            if re.search("// 8P\. *", line):
                self.pot_8P =\
                    float(line.split()[2])
        self.data[:,[1,2]] *= self.iterations
        #self.data[:,[1,2]] /= self.integration
        #self.data[:,0] += self.pot_8P
    def merge(self, scan):
        self.data = np.vstack((self.data, scan.data))

