import numpy as np
import pybullet as pb
import pybullet_data
from scipy.spatial.transform import Rotation as R

def read_parameters(dbg_params):
    '''Reads values from debug parameters

    Parameters
    ----------
    dbg_params : dict
        Dictionary where the keys are names (str) of parameters and the values are
        the itemUniqueId (int) for the corresponing debug item in pybullet

    Returns
    -------
    dict
        Dictionary that maps parameter names (str) to parameter values (float)
    '''
    values = dict()
    for name, param in dbg_params.items():
        values[name] = pb.readUserDebugParameter(param)

    return values

def interactive_transformations(axis_id):
    np.set_printoptions(suppress=True, precision=4)

    dbg = dict()
    # for view matrix
    translation_params = ['x', 'y', 'z']
    for p in translation_params:
        dbg[p] = pb.addUserDebugParameter(p, -10, 10, 0)

    rotation_params = ['roll', 'pitch', 'yaw']
    for p in rotation_params:
        dbg[p] = pb.addUserDebugParameter(p, -180, 180, 0)

    dbg['print'] =  pb.addUserDebugParameter('print details', 1, 0, 1)
    old_print_val = 1
    while 1:
        values = read_parameters(dbg)

        pos = np.array([values[c] for c in translation_params])
        euler = np.radians([values[c] for c in rotation_params])

        #update visual aids for camera, target pos
        quat = np.array(pb.getQuaternionFromEuler(euler))
        pb.resetBasePositionAndOrientation(axis_id, pos, quat)

        T_mtx = np.eye(4)
        T_mtx[:3,:3] = R.from_quat(quat).as_matrix()
        T_mtx[:3,3] = pos

        if old_print_val != values['print']:
            old_print_val = values['print']
            print("\n========================================")
            print(f"translation = {pos}")
            print(f"euler angles = {euler}")
            print(f"quat = {quat}")
            T_mtx_str = f"{T_mtx}".splitlines()
            print(f"T_mtx = {T_mtx_str[0]}")
            print(f"        {T_mtx_str[1]}")
            print(f"        {T_mtx_str[2]}")
            print(f"        {T_mtx_str[3]}")
            print("========================================\n")

if __name__ == "__main__":
    pb.connect(pb.GUI)

    # add axis
    axis_id = pb.loadURDF('assets/urdf/axis.urdf', globalScaling=0.5)

    interactive_transformations(axis_id)
