import numpy as np
from scipy.integrate import odeint
from .rate import dict2Params as P

class BaseModel:
    def set_params(self, params):
        self.params = P(params)
        return self



class Decay(BaseModel):
    params = P({"N0": 100, "r": 10})

    @staticmethod
    def func (p, x):
        N0 = p["N0"].value
        r  = p["r" ].value
        return (N0*np.exp(-x*r), )
 


class Change(BaseModel):
    params = P({"N0": 1000, "N1": 10, "r": 1, "loss":0})
    params["loss"].set(vary=False)

    @staticmethod
    def func(p, x):
        N0 = p["N0"].value
        N1 = p["N1"].value
        r = p["r"].value
        loss = p["loss"].value
        return (
            np.exp(-x*r)*N0,
            np.exp(-x*loss)*(N1 + N0*r/(loss-r)*(np.exp(-x*(r-loss))-1))
            )


class ChangeChannel(BaseModel):
    params = P({"N0": 1000, "N1": 100, "r": 10, "bratio":.5, "loss":0})
    params["loss"].set(vary=False)

    @staticmethod
    def func(p, x):
        N0 = p["N0"].value
        N1 = p["N1"].value
        r = p["r"].value
        bratio = p["bratio"].value
        loss = p["loss"].value
        return (
            np.exp(-x*r)*N0,
            np.exp(-x*loss)*(N1 + bratio*N0*r/(loss-r)*(np.exp(-x*(r-loss))-1))
            )   


class ChangeChannelBG(BaseModel):
    params = P({"N0": 1000, "N1": 100, "r": 10, "bratio":.5, "bg": 10})

    @staticmethod
    def func(p, x):
        N0 = p["N0"].value
        N1 = p["N1"].value
        r = p["r"].value
        bratio = p["bratio"].value
        bg = p["bg"].value
        return (
            np.exp(-x*r)*N0 + bg,
            N0*bratio*(1-np.exp(-x*r)) + N1
            )


class Change2Channel(BaseModel):
    params = P({
        "N0" : 1000.,      "N1": 100.,
        "N2" : 1.,         "r1": 1.,
        "r2" : 10.,        "r":20})

    @staticmethod
    def func (p, x):
        N0 = p["N0"].value
        N1 = p["N1"].value
        N2 = p["N2"].value
        r1 = p["r1"].value
        r2 = p["r2"].value
        r = p["r"].value
        return (
            np.exp(-x*r)*N0,
            N0*r1/r*(1-np.exp(-x*r)) + N1,
            N0*r2/r*(1-np.exp(-x*r)) + N2,
            )

class ChangeNChannel(BaseModel):
    params = P({
        "N0" : 1000.,        "r0":20,
        "N1": 100.,         "r1": 1.,
        "N2" : 1.,         "r2": 10.})

    def __init__(self, N):
        self.N = N

    #@staticmethod
    def func (self, p, x):
        N0 = p["N0"].value
        r0 = p["r0"].value
        return [np.exp(-x*r0)*N0] +\
                [N0*p["r%d"%i].value/r0*(1-np.exp(-x*r0)) + p["N%d"%i].value for i in range(1, self.N)]


class Equilib(BaseModel):
    params = P({
        "N0" : 1000.,      "N1": 100.,
        "r0" : 1.,         "r1": 1.})

    @staticmethod
    def func(p, x):
        N0 = p["N0"].value
        N1 = p["N1"].value
        r0 = p["r0"].value
        r1 = p["r1"].value
        r = r0+r1
        C2 = (N0 + N1)/(1+r0/r1)
        C1 = N0 - C2
        return (
            C1*np.exp(-x*r) + C2,
            -C1*np.exp(-x*r) + C2*r0/r1
            )

class CPlusPlus(BaseModel):
    params = P({
        "Cpp": 100.,       "rC":1.,
        "C": 10.,          "rX":1.,
        "X": 1.,           "rH3":1.,
        "H3": 1.,           "rH5": 1.
        })

    @staticmethod
    def func(p, x):
        specnames = ["Cpp", "C", "X", "H3"]
        ratenames = ["rC", "rX", "rH3", "rH5"]

        p = p.valuesdict()
        rC, rX, rH3, rH5 = [p[name] for name in ratenames]
        Cpp, C, X, H3 = range(4)
        eqn = lambda y, x: [\
                # C++
                -(rC+rX)*y[Cpp],\
                # C+
                rC*y[Cpp],\
                 # H+ or H2+...
                rX*y[Cpp] - rH3*y[X],\
                 # H3
                rH3*y[X]- rH5*y[H3],\
                ]
        y0 = [p[name] for name in specnames]
        t = np.r_[0, x]
        y = odeint(eqn, y0, t, mxstep=10000)
        #y[:,C] /= p["H3disc"]
        y[:,H3] *= p["H3disc"]
        res = y[1:,[Cpp, C, H3]]
        return res.T


class Ar(BaseModel):
    params = P({
        "Ar": 100.,         "r1":1.,
        "ArH": 10.,         "r2":1.,
        "H2": 1.,           "r3":1.,
        "H3": 1.,           "r4":1.,})

    @staticmethod
    def func(p, x):
        specnames = ["Ar", "ArH", "H2", "H3"]
        ratenames = ["r1", "r2", "r3", "r4"]

        p = p.valuesdict()
        r1, r2, r3, r4 = [p[name] for name in ratenames]
        Ar, ArH, H2, H3 = range(4)
        eqn = lambda y, x: [\
                # Ar
                -(r1+r2)*y[Ar],\
                # ArH
                r1*y[Ar] - r3*y[ArH],\
                 # H2
                r2*y[Ar] - r4*y[H2],\
                 # H3
                r3*y[ArH] + r4*y[H2],\
                ]
        y0 = [p[name] for name in specnames]
        t = np.r_[0, x]
        y = odeint(eqn, y0, t, mxstep=10000)
        y[:,ArH] *= p["ArHdisc"]
        y[:,H2] *= p["H2disc"]
        y[:,H3] *= p["H3disc"]
        res = y[1:,:5]
        return res.T

class Ar_simple(BaseModel):
    params = P({
        "Ar": 100.,         "r1":1.,
        "ArH": 10.,         "r2":1.,
        "H2": 1.,           "r3":1.,
                             "r4":1.,})

    @staticmethod
    def func(p, x):
        specnames = ["Ar", "ArH", "H2"]
        ratenames = ["r1", "r2", "r3", "r4"]

        p = p.valuesdict()
        r1, r2, r3, r4 = [p[name] for name in ratenames]
        Ar, ArH, H2 = range(3)
        eqn = lambda y, x: [\
                # Ar
                -(r1+r2)*y[Ar],\
                # ArH
                r1*y[Ar] - r3*y[ArH],\
                 # H2
                r2*y[Ar] - r4*y[H2],\
                ]
        y0 = [p[name] for name in specnames]
        t = np.r_[0, x]
        y = odeint(eqn, y0, t, mxstep=10000)
        y[:,ArH] *= p["ArHdisc"]
        y[:,H2] *= p["H2disc"]
        res = y[1:,]
        return res.T

class Oplus(BaseModel):
    params = P({
        "O": 100.,          "OH": 10.,
        "OH2": 1.,          "OH3": 1.,
        "H": 10.,
        "rOH":1.,          "rOH2":10.,
        "rO3":1,            "rOH3":10,
        "rO3d":1})

    @staticmethod
    def func(p, x):
        specnames = ["O", "OH", "OH2", "OH3", "H"]
        ratenames = ["rOH", "rOH2", "rH", "rOH3", "rHd"]

        p = p.valuesdict()
        rOH, rOH2, rH, rOH3, rHd = [p[name] for name in ratenames]
        O, OH, OH2, OH3, H = range(5)
        eqn = lambda y, x: [\
                # O+
                -(rOH + rH)*y[O],\
                # OH+
                rOH*y[O] - rOH2*y[OH],\
                # OH2+
                rOH2*y[OH] - rOH3*y[OH2],\
                # OH3+
                rOH3*y[OH2],\
                # H+
                rH*y[O] - rHd*y[H],
                ]
        y0 = [p[name] for name in specnames]
        t = np.r_[0, x]
        y = odeint(eqn, y0, t, mxstep=10000)
        y[:,H]   *= p["Hdisc"]
        y[:,OH]  *= p["OHdisc"]
        y[:,OH2] *= p["OH2disc"]
        y[:,OH3] *= p["OH3disc"]
        res = y[1:,:5]
        return res.T


class OminusHD(BaseModel):
    params = P({
        "O": 100.,   "OH": 10.,
        "OD": 1.,    "r5": .1,
        "r1":1.,     "r4":10.,
        "r2":1,      "r3":1})

    @staticmethod
    def func(p, x):
        specnames = ["O", "OH", "OD"]
        ratenames = ["r1", "r2", "r3", "r4", "r5"]

        p = p.valuesdict()
        r1, r2, r3, r4, r5 = [p[name] for name in ratenames]
        O, OH, OD = range(3)
        eqn = lambda y, x: [\
                # O-
                -(r1+r2+r3)*y[O],\
                # OH-
                r2*y[O] - r4*y[OH] + r5*y[OD],\
                # OD-
                r3*y[O] + r4*y[OH] - r5*y[OD],
                ]
        y0 = [p[name] for name in specnames]
        t = np.r_[0, x]
        y = odeint(eqn, y0, t, mxstep=10000)
#        y[:,H]   *= p["Hdisc"]
#        y[:,OH]  *= p["OHdisc"]
#        y[:,OH2] *= p["OH2disc"]
#        y[:,OH3] *= p["OH3disc"]
        res = y[1:]
        return res.T


class OHplus(BaseModel):
    params = P({
        "O": 100.,                    "OH": 10.,
        "OH2": 1.,          "OH3": 1.,
        "rOH":1.,          "rOH2":10.,
                    "rOH3":10,
        })

    @staticmethod
    def func(p, x):
        specnames = ["O", "OH", "OH2", "OH3", "NH3"]
        ratenames = ["rOH", "rOH2", "rOH3", "rNH4"]

        p = p.valuesdict()
        rOH, rOH2, rOH3, rNH4 = [p[name] for name in ratenames]
        O, OH, OH2, OH3, NH3 = range(5)
        eqn = lambda y, x: [\
                # O+
                -(rOH)*y[O],\
                # OH+
                rOH*y[O] - rOH2*y[OH],\
                # OH2+
                rOH2*y[OH] - rOH3*y[OH2],\
                # OH3+
                rOH3*y[OH2],
                # NH3+ TODO add NH4+
                -rNH4*y[NH3]
                ]
        y0 = [p[name] for name in specnames]
        t = np.r_[0, x]
        y = odeint(eqn, y0, t, mxstep=10000)
        y[:,OH] += y[:,NH3] # unresolved masses
        y[:,OH2] *= p["OH2disc"]
        y[:,OH3] *= p["OH3disc"]
        res = y[1:,:4]
        return res.T


class NH(BaseModel):
    params = P({
        "N": 100.,          "NH": 10.,
        "NH2": 1.,          "NH3": 1.,
        "H3": 10.,          "N15":10.,
        "rNH":1.,          "rNH2":10.,
        "rH3":1,            "rNH3":10,
        "rH3d":1})

    @staticmethod
    def func(p, x):
        specnames = ["N", "NH", "NH2", "NH3", "H3", "N15"]
        ratenames = ["rNH", "rNH2", "rH3", "rNH3", "rH3d"]

        p = p.valuesdict()
        rNH, rNH2, rH3, rNH3, rH3d = [p[name] for name in ratenames]
        N, NH, NH2, NH3, H3, N15 = range(6)
        eqn = lambda y, x: [\
                # N+
                -rNH*y[N],\
                # NH+
                -(rNH2 + rH3)*y[NH] + rNH*y[N],\
                # NH2+
                rNH2*y[NH] - rNH3*y[NH2],\
                # NH3+
                rNH3*y[NH2],\
                # H3+
                rH3*y[NH] - rH3d*y[H3],\
                # 15N+
                -rNH*y[N15],
                ]
        y0 = [p[name] for name in specnames]
        t = np.r_[0, x]
        y = odeint(eqn, y0, t, mxstep=10000)
        y[:,H3] *= p["H3disc"]
        y[:,NH2] *= p["NH2disc"]
        y[:,NH3] *= p["NH3disc"]
        res = y[1:,:5]
        res[:,1] += y[1:,N15] # add the relaxed and excite NH3+
        return res.T

class NplusHD(BaseModel):
    params = P({
        "N": 100.,          "NH": 1.,
        "ND": 1.,           "NH2": 1.,
        "rNH":1.,           "rND":1,
        "rNH2":50,          "rH3":1,
        "rNH3":10,          "rNDd":50,})

    @staticmethod
    def func(p, x):
        specnames = ["N", "NH", "ND", "NH2"]
        ratenames = ["rNH", "rND", "rNH2", "rH3", "rNH3", "rNDd"]

        p = p.valuesdict()
        rNH, rND, rNH2, rH3, rNH3, rNDd = [p[name] for name in ratenames]
        N, NH, ND, NH2 = range(4)
        eqn = lambda y, x: [\
                # N+
                -(rNH+rND)*y[N],\
                # NH+
                -(rNH2 + rH3)*y[NH] + rNH*y[N],\
                # ND+
                -rNDd*y[ND] + rND*y[N],\
                # NH2+
                rNH2*y[NH] - rNH3*y[NH2],\
                ]
        y0 = [p[name] for name in specnames]
        t = np.r_[0, x]
        y = odeint(eqn, y0, t, mxstep=10000)
        y[:,ND] +=  y[:,NH2]
        res = y[1:,:3]
        return res.T

class NHn_long(BaseModel):
    params = P({
        "N": 100.,          "NH": 10.,
        "NH2": 1.,          "NH3": 1.,
        "NH4": .1,          "H3": 10.,
        "rNH":1.,           "rNH2":10.,
        "rH3":1,            "rNH3":10,
        "rNH4":1,           "rH3d":1,
        "rNH3rel":1,        "rNH4exc":10})

    @staticmethod
    def func(p, x):
        N, NH, NH2, NH3, NH4, H3, NH3e = range(7)
        p = p.valuesdict()
        eqn = lambda y, x: [\
                # N+
                -p["rNH"]*y[N],\
                # NH+
                p["rNH"]*y[N] - p["rH3"]*y[NH] - p["rNH2"]*y[NH],\
                # NH2+
                p["rNH2"]*y[NH] - p["rNH3"]*y[NH2],\
                # NH3+ relaxed
                p["rNH3rel"]*y[NH3e] - p["rNH4"]*y[NH3] - p["NH3loss"]*y[NH3],\
                # NH4+
                p["rNH4"]*y[NH3] + p["rNH4exc"]*y[NH3e],\
                # H3+
                p["rH3"]*y[NH] - p["rH3d"]*y[H3],\
                # excited NH3+
                p["rNH3"]*y[NH2] - p["rNH3rel"]*y[NH3e] - p["rNH4exc"]*y[NH3e],\
                ]
        y0 = [p["N"], p["NH"], p["NH2"], p["NH3"], p["NH4"], p["H3"], 0.]
        t = np.r_[0, x]
        y = odeint(eqn, y0, t, mxstep=10000)
        res = y[1:,[0,1,2,3,4,5]]
        res[:,3] += y[1:,NH3e] # sum the relaxed and excited NH3+
        #res *= np.array([1] + list(disc))
        res[:,H3] *= p["H3disc"]
        return res.T


class NHn_short(BaseModel):
    params = P({
        "N":400.,       "NH":.1,
        "NH2":.1,       "NH3":.1,
        "H3":.1,        "rNH":10.,
        "rNH2":100.,    "rNH3":100.,
        "rNH4":10.,     "rH3":10,
        "rH3d":10.
        })

    @staticmethod
    def func(p, x):
        N, NH, NH2, NH3, H3, NH4 = range(6)
        p = p.valuesdict()
        eqn = lambda y, x: [\
                # N+
                -p["rNH"]*y[N],\
                # NH+
                p["rNH"]*y[N] - p["rH3"]*y[NH] - p["rNH2"]*y[NH],\
                # NH2+
                p["rNH2"]*y[NH] - p["rNH3"]*y[NH2],\
                # NH3+ relaxed
                p["rNH3"]*y[NH2] - p["rNH4"]*y[NH3],\
                # H3+
                p["rH3"]*y[NH] - p["rH3d"]*y[H3],\
                ]
        y0 = [p["N"], p["NH"], p["NH2"], p["NH3"], p["H3"]]
        t = np.r_[0, x]
        y = odeint(eqn, y0, t, mxstep=10000)
        res = y[1:,[0,1,2,3,4]]
        res[:,H3] *= p["H3disc"]
        return res.T
