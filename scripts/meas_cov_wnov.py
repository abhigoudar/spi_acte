import numpy as np
import cvxpy as cp
import defs
import process_models_sym
import measurement_models_sym

VERBOSE = True

def compute_J(J_n, proc_model, meas_model, R_i, dt_i : float):
    H = meas_model.jacobian([])
    D_11 = proc_model.FT_Qi_F(dt_i) + np.dot(H.T, np.dot(R_i, H))
    D_22 = proc_model.Qi(dt_i)
    D_12 = proc_model.FT_Qi(dt_i)
    D_21 = D_12.T
    J_n1 = D_22 - (D_21 @ (np.linalg.inv(J_n + D_11) @ D_12))
    return J_n1

def solve_for_R(proc_model, meas_model, ka_, dti : float, *args):
    #
    print("Desired accuracy:",ka_, " meas freq.", dti)
    ka = ka_**2
    #
    state_dim = proc_model.state_dim
    pos_dim = proc_model.pos_dim
    #
    if VERBOSE:
        print("State dim:[%d] pos dim:[%d]"%(state_dim, pos_dim))

    R_i_est = cp.Variable((2,2), PSD=True)
    J_ss = cp.Variable((2,2), PSD=True)
    #
    if state_dim > pos_dim:
        J_des = np.zeros((state_dim, state_dim), dtype=float)
        J_des[:pos_dim, :pos_dim] = 1.0/ka * np.eye(pos_dim, dtype=float)
    else:
        J_des = 1.0/ka * np.eye(pos_dim, dtype=float)

    if VERBOSE:
        print("Max required evalue:", np.max(np.sqrt(np.linalg.inv(J_des[:pos_dim,:pos_dim]))))
    #
    H = meas_model.jacobian(*args)
   #
    D_11 = proc_model.FT_Qi_F(dti)
    D_22 = H.T @ (R_i_est @ H)
    D_12 = -1 * proc_model.FT_Qi(dti)
    D_21 = -1 * proc_model.FT_Qi(dti).T
    Q_i = proc_model.Qi(dti)
    #
    C1 = cp.bmat([[J_des + D_11 + D_22,     D_12],
                [D_21,            Q_i - J_des]])
    if VERBOSE:
        print("Is constraint DCP", C1.is_dcp())

    objective = cp.Minimize(cp.trace(R_i_est))
    constraints = [C1 >> 0]

    if VERBOSE:
        print('\nSolving optimization problem')

    problem = cp.Problem(objective, constraints)

    if(problem.is_dcp()):

        if VERBOSE:
            print("Is problem DCP ->", problem.is_dcp())

        problem.solve(solver="MOSEK", verbose=True)
    else:
        print("Problem is not DCP.")
        ## TODO: throw exception
    # #
    if VERBOSE:
        print("Solution status:[%s]"%problem.status)
    #
    if not problem.status == "optimal":
        return problem.status, []
    #
    R_est = np.linalg.inv(R_i_est.value)

    if VERBOSE:
        print("Solution status:[%s]"%problem.status)
        print("Measurement uncertainty: \n R_i \n", R_i_est.value, "\n  R: \n", R_est)
    #
    try:
        cov_ss = np.linalg.inv(J_ss.value[:pos_dim,:pos_dim])
        # print("Desired cov ss:\n", ka)
        if VERBOSE:
            print("Estimated cov ss:\n", cov_ss)
            print(" Max obtained (sqrt) evalue for E_ss:%.3f\n"%np.sqrt(np.max(np.linalg.eigvals(cov_ss))))
    except:
        print("Value of J_ss unavailable\n")
    #
    return problem.status, R_est

if __name__ == "__main__":
    proc_model = process_models_sym.process_model_factory(defs.PROCESS_MODEL)
    meas_model = measurement_models_sym.measurement_model_factory(defs.PROCESS_MODEL, defs.MEASUREMENT_MODEL)
    #
    #
    meas_freq = 20.0
    print("\n \n ---------")
    print("\nCalculating meas cov for [%s] with [%s] dti:[%.3f]"%(defs.PROCESS_MODEL, defs.MEASUREMENT_MODEL, meas_freq))
    status, R_ = solve_for_R(proc_model, meas_model, defs.DESIRED_ACCURACY, meas_freq)
    print('Problem status: [{}] \n R:\n'.format(status), R_)
