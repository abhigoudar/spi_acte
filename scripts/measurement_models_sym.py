#!/usr/bin/env python3
import numpy as np
from defs import *

class MeasurementModel():
    def __init__(self, cov, **kwargs):
        self.cov = cov
    def jacobian(self, *args):
        return []

class PositionMeasurementModelRn(MeasurementModel):
    def __init__(self, cov: np.array, **kwargs):
        n = np.shape(cov)[0]
        super().__init__(cov)
        self.dim = n
        self.inv_cov = np.linalg.inv(cov)

    def jacobian(self, *args):
        return np.eye(self.dim, dtype=float)

class PositionMeasurementModelSEn(MeasurementModel):
    def __init__(self, cov: np.array, **kwargs):
        n = np.shape(cov)[0]
        super().__init__(cov)
        self.dim = n
        self.inv_cov = np.linalg.inv(cov)

    def jacobian(self, *args):
        return np.eye(self.dim, dtype=float)

class RangeMeasurementModelRn(MeasurementModel):
    def __init__(self, cov: float, **kwargs):
        super().__init__(cov)
        self.inv_cov = 1.0/cov
        self.dim = 1
        if 'nominal_jacobian' in kwargs:
            self.nom_jac = kwargs['nominal_jacobian']
            n_m = self.nom_jac.shape[0]
            self.inv_cov = np.eye(n_m) * 1.0/cov
        else:
            self.nom_jac = []
    '''
    @brief jacobian for range measurement
    @args expects anchor position, robot position
    optionally robot orientation, lever arm
    '''
    def jacobian(self, *args):
        if len(args) == 2:
            anc_pos = args[0]
            rob_pos = args[1]
            assert len(anc_pos) == len(rob_pos)
            dp = anc_pos - rob_pos
            return (dp * -1.0/np.linalg.norm(dp))
        else:
            return self.nominal_jacobian()

    def nominal_jacobian(self, *args):
        return self.nom_jac
    
def measurement_model_factory(proc_mode, meas_mode):
    if meas_mode == 'POSITION':
        if 'WNOV' in proc_mode:
            if 'R2' or 'R3' in proc_mode:
                dim = int(PM_PARAMS[proc_mode]['dim'])
                R_ = np.eye(dim, dtype=float) * MM_PARAMS[meas_mode]['R']
                return PositionMeasurementModelRn(R_)
        if 'WNOA' in proc_mode:
            if 'R2' or 'R3' in proc_mode:
                dim = int(PM_PARAMS[proc_mode]['dim'])
                R_ = np.eye(dim, dtype=float) * MM_PARAMS[meas_mode]['R']
                return PositionMeasurementModelRn(R_)


    elif meas_mode == 'RANGE':
        R_ = np.eye(1, dtype=float) * MM_PARAMS[meas_mode]['R']
        params = {}
        params['nominal_jacobian'] = MM_PARAMS[meas_mode]['H']
        return RangeMeasurementModelRn(R_, **params)