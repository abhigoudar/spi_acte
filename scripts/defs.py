# Motion model parameters
import numpy as np

# Time duration of robot trajectory
TIME_TOTAL = 120 #seconds

DESIRED_ACCURACY = 0.02 # (m)

PROCESS_MODEL="WNOV_R2"

# Options for Measurement model

# MEASUREMENT_MODEL="POSITION"
MEASUREMENT_MODEL="RANGE"

# Parameters for motion model
PM_PARAMS = {
    ## Parameters for white-noise-on-velocity motion model (simulation)
    'WNOV_R2' : {
        'Q' : np.array([[0.001, 0,],
                        [0, 0.001]]),
        'dim'       : 2,
        'state_dim' : 2, # position
        'input_dim' : 2,
    },
    ## Parameters for white-noise-on-velocity motion model (real experiments)
    # 'WNOV_R2' : {
    #     'Q' : np.array([[0.02, 0,],
    #                     [0, 0.02]]),
    #     'dim'       : 2,
    #     'state_dim' : 2, # position
    #     'input_dim' : 2,
    # },
}

# Parameters for measurement model
MM_PARAMS = {
    'POSITION' : {
        'R' : 0.01,
    },
    'RANGE' : {
        'R' : 0.0025,
        # For range measurement model we set the jacobian here
        'H' : np.array([[0.70710678, 0.70710678],
                        [0.70710678, -0.70710678]]),
    },
}

CHECK_DUALITY_GAP = True