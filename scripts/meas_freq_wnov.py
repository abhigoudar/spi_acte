import numpy as np
import cvxpy as cp
import defs
import process_models_sym
import measurement_models_sym

VERBOSE = True

def get_meas_model_matrices(meas_model, *args):
    H = meas_model.jacobian(*args)
    R_i = meas_model.inv_cov
    return H, R_i

def compute_J(J_n, proc_model, meas_model, dt_i : float):
    H, R_i = get_meas_model_matrices(meas_model, [])
    #
    D_11 = proc_model.FT_Qi_F(dt_i) + np.dot(H.T, np.dot(R_i, H))
    D_22 = proc_model.Qi(dt_i)
    #
    D_12 = proc_model.FT_Qi(dt_i)
    D_21 = D_12.T
    J_n1 = D_22 - (D_21 @ (np.linalg.inv(J_n + D_11) @ D_12))
    return J_n1

def solve_for_fm(proc_model, meas_model, ka_, *args):
    #
    print("\n \n ---------")
    print("Desired accuracy:",ka_)
    ka = ka_**2
    #
    state_dim = proc_model.state_dim
    pos_dim = proc_model.pos_dim

    if VERBOSE:
        print("State dim:[%d] pos dim:[%d]"%(state_dim, pos_dim))

    dt_i = cp.Variable(nonneg=False)
    J_ss = cp.Variable((state_dim,state_dim), PSD=True)
    #
    if state_dim > pos_dim:
        J_des = np.zeros((state_dim, state_dim), dtype=float)
        J_des[:pos_dim, :pos_dim] = 1.0/ka * np.eye(pos_dim, dtype=float)
    else:
        J_des = 1.0/ka * np.eye(pos_dim, dtype=float)

    if VERBOSE:
        print("Max required evalue:", np.max(np.sqrt(np.linalg.inv(J_des[:pos_dim,:pos_dim]))))
    #
    H, R_i = get_meas_model_matrices(meas_model, [])

    if VERBOSE:
        print("Shape of H:", np.shape(H), H)
        print("Shape of R_i:", np.shape(R_i), R_i)
    #
    D_11 = proc_model.FT_Qi_F(dt_i)
    D_22 = np.dot(H.T, np.dot(R_i, H))
    D_12 = -1 * proc_model.FT_Qi(dt_i)
    D_21 = -1 * proc_model.FT_Qi(dt_i).T
    Q_i = proc_model.Qi(dt_i)
    #
    C1 = cp.bmat([[J_des + D_11 + D_22,     D_12],
                [D_21,            Q_i - J_des]])

    if VERBOSE:
        print("Is constraint DCP", C1.is_dcp())
    #
    objective = cp.Minimize(dt_i)
    constraints = [C1 >> 0]

    if VERBOSE:
        print('\nSolving optimization problem')

    problem = cp.Problem(objective, constraints)
    if(problem.is_dcp()):
        if VERBOSE:
            print("Is problem DCP ->", problem.is_dcp())
        problem.solve(verbose=True)
    else:
        print("Problem is not DCP.")
    # #
    if VERBOSE:
        print("Solution status:[%s]"%problem.status)
        print("Measurement frequency:%f"%dt_i.value, "  dt:%f"%(1.0/dt_i.value))
    #
    J_n1 = compute_J(J_des, proc_model, meas_model, dt_i.value)
    cov_n1 = np.linalg.inv(J_n1[:pos_dim,:pos_dim])
    if VERBOSE:
        print("\n Max obtained (sqrt) evalue for E_(n+1):%.3f"%np.sqrt(np.max(np.linalg.eigvals(cov_n1))))
    try:
        cov_ss = np.linalg.inv(J_ss.value[:pos_dim,:pos_dim])

        if VERBOSE:
            print(" Max obtained (sqrt) evalue for E_ss:%.3f\n"%np.sqrt(np.max(np.abs(np.linalg.eigvals(cov_ss)))))
    except:
        print("Value of J_ss unavailable\n")

    if defs.CHECK_DUALITY_GAP:
        #
        temp = np.bmat([[J_des + D_22, np.zeros((2,2), dtype=float)], [np.zeros((2,2), dtype=float), - J_des]])
        d_star = -1 * np.trace(temp @ constraints[0].dual_value)
        p_star = problem.value
        if VERBOSE:
            print('Dual value (d*):', d_star)
            print('Primal value (p*):', p_star)
        duality_gap = p_star - d_star
    else:
        duality_gap = -1
    #
    return problem.status, dt_i.value, J_ss.value, duality_gap

if __name__ == "__main__":
    proc_model = process_models_sym.process_model_factory(defs.PROCESS_MODEL)
    meas_model = measurement_models_sym.measurement_model_factory(defs.PROCESS_MODEL, defs.MEASUREMENT_MODEL)
    print("Calculating dt_i for [%s] with [%s]"%(defs.PROCESS_MODEL, defs.MEASUREMENT_MODEL))
    #
    status, freq, j_ss, d_gap = solve_for_fm(proc_model, meas_model, defs.DESIRED_ACCURACY, [])
    print('status:{} freq:{} d_gap:{}'.format(status, freq, d_gap))
