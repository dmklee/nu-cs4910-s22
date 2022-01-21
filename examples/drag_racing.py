import time
import numpy as np
import pybullet as pb
import pybullet_data

START_POS = (0,0,0.2)
RACE_LENGTH = 5
GRAVITY = -9.8
MAX_FORCE = 12

FRONT_MOTOR_IDS = [2,3]
REAR_MOTOR_IDS = [0,1]

def init_sim(gravity=-10):
    client = pb.connect(pb.GUI)
    pb.setGravity(0,  0, gravity)

    # configure camera to look down race course
    pb.resetDebugVisualizerCamera(cameraDistance=1.8,
                                  cameraYaw=-85.2,
                                  cameraPitch=-26.6,
                                  cameraTargetPosition=(0.27, -0.16, 0.67))

    # add plane
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane = pb.loadURDF("plane.urdf")
    pb.changeDynamics(plane, -1, rollingFriction=0.05)

    # add car
    car = pb.loadURDF('assets/urdf/racecar.urdf', START_POS)
    for rear_wheel_id in REAR_MOTOR_IDS:
        pb.changeDynamics(car, rear_wheel_id,
                          spinningFriction=0.5)

    # allow front wheels to spin freely
    pb.setJointMotorControlArray(bodyUniqueId=car,
                                 jointIndices=FRONT_MOTOR_IDS,
                                 controlMode=pb.VELOCITY_CONTROL,
                                 forces=2*[0])

    # add finish line
    pb.loadURDF('assets/urdf/race_finish_line.urdf', (RACE_LENGTH, 0, 0), useFixedBase=True )

    return car

def reset_car(car):
    # reset car location
    pb.resetBasePositionAndOrientation(car, START_POS, (0,0,0,1))
    pb.resetBaseVelocity(car, np.zeros(3), np.zeros(3))

    # set rear motors to torque-limited acceleration
    pb.setJointMotorControlArray(bodyUniqueId=car,
                                 jointIndices=REAR_MOTOR_IDS,
                                 controlMode=pb.VELOCITY_CONTROL,
                                 targetVelocities=2*[10000],
                                 forces=2*[MAX_FORCE])

def apply_downforce(car, magnitude):
    pb.applyExternalForce(car, -1,
                          forceObj=(0,0, -magnitude),
                          posObj=(0,0,0),
                          flags=pb.LINK_FRAME)

def main(car):
    dbg_button =  pb.addUserDebugParameter('begin race', 1, 0, 1)
    old_button_val = 2

    is_racing = True
    while 1:
        new_button_val = pb.readUserDebugParameter(dbg_button)
        if new_button_val != old_button_val:
            old_button_val = new_button_val
            reset_car(car)
            start_time = time.time()
            is_racing = True

        if is_racing:
            # check if car body is fully past finish line
            aabb = pb.getAABB(car, 0)
            if aabb[0][0] > RACE_LENGTH:
                finish_time = time.time()
                is_racing = False
                elapsed_time = finish_time - start_time
                print(f'Lap Time : {elapsed_time:.3f}s')

            pb.stepSimulation()
            time.sleep(1./256)


if __name__ == "__main__":
    car = init_sim(GRAVITY)

    # print out joint and link ids
    # num_joints = pb.getNumJoints(car)
    # links = dict()
    # joints = dict()
    # for i in range(num_joints):
        # j_info = pb.getJointInfo(car, i)
        # j_id = j_info[0]
        # j_name = j_info[1].decode('UTF-8')
        # j_type = j_info[2]
        # link_name = j_info[12].decode('UTF-8')
        # parent_id = j_info[16]

        # print(f"Joint{j_id} = {j_name}, {link_name}")

    main(car)
