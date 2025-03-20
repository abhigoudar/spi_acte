from defs import *
import numpy as np

class MotionModel:
    def __init__(self, Q_):
        self.id = -1
        self.Q = Q_

'''
White-noise-on-velocity motion model on 2D position only
'''
class WNOVMotionModelR2Sym(MotionModel):
    def __init__(self, Q_ : np.array):
        assert np.shape(Q_) == (2,2), " Size of Q_ should be (2,2)"
        self.Q = Q_
        self.Q_inv = np.linalg.inv(self.Q)
        self.state_dim = 2
        self.pos_dim = 2

    def FT_Qi_F(self, *args):
        dti = args[0]
        return self.Qi(dti)

    def FT_Qi(self, *args):
        dti = args[0]
        return self.Qi(dti)

    def Qi(self, *args):
        dti = args[0]
        Q_ = dti * self.Q_inv
        return Q_

    
def process_model_factory(proc_mode):
    Q_ = PM_PARAMS[proc_mode]['Q']
    print('Q_:', Q_)
    if proc_mode == "WNOV_R2":
        return WNOVMotionModelR2Sym(Q_)
    else:
        return None