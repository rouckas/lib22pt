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
        "r2" : 10.})

    @staticmethod
    def func (p, x):
        N0 = p["N0"].value
        N1 = p["N1"].value
        N2 = p["N2"].value
        r1 = p["r1"].value
        r2 = p["r2"].value
        r = r1+r2
        return (
            np.exp(-x*r)*N0,
            N0*r1/r*(1-np.exp(-x*r)) + N1,
            N0*r2/r*(1-np.exp(-x*r)) + N2,
            )


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
