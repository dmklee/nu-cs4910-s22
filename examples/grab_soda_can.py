from typing import List
import time
import numpy as np
import pybullet as pb
import pybullet_data


class RobotArm:
    GRIPPER_CLOSED = 0.
    GRIPPER_OPENED = 1.
    def __init__(self):
        self._id = pb.loadURDF("assets/urdf/xarm.urdf",
                               basePosition=(0, 0, 0.012),
                               flags=pb.URDF_USE_SELF_COLLISION)
        # these are hard coded based on how urdf is written
        self.arm_joint_ids = [1,2,3,4,5]
        self.gripper_joint_ids = [6,7]
        self.dummy_joint_ids = [8]
        self.finger_joint_ids = [9,10]
        self.end_effector_link_index = 11

        self.arm_joint_limits = np.array(((-2, -1.58, -2, -1.8, -2),
                                          ( 2,  1.58,  2,  2.0,  2)))
        self.gripper_joint_limits = np.array(((0.05,0.05),
                                              (1.38, 1.38)))

        # chosen to move arm out of view of camera
        self.home_arm_jpos = [0., -1.1, 1.4, 1.3, 0.]

        # joint constraints are needed for four-bar linkage in xarm fingers
        for i in [0,1]:
            constraint = pb.createConstraint(self._id,
                                             self.gripper_joint_ids[i],
                                             self._id,
                                             self.finger_joint_ids[i],
                                             pb.JOINT_POINT2POINT,
                                             (0,0,0),
                                             (0,0,0.03),
                                             (0,0,0))
            pb.changeConstraint(constraint, maxForce=1000000)

        # reset joints in hand so that constraints are satisfied
        hand_joint_ids = self.gripper_joint_ids + self.dummy_joint_ids + self.finger_joint_ids
        hand_rest_states = [0.05, 0.05, 0.055, 0.0155, 0.031]
        [pb.resetJointState(self._id, j_id, jpos)
                 for j_id,jpos in zip(hand_joint_ids, hand_rest_states)]

        # allow finger and linkages to move freely
        pb.setJointMotorControlArray(self._id,
                                     self.dummy_joint_ids+self.finger_joint_ids,
                                     pb.POSITION_CONTROL,
                                     forces=[0,0,0])

    def move_gripper_to(self, position):
        n_joints = pb.getNumJoints(self._id)
        all_jpos = pb.calculateInverseKinematics(self._id,
                                                 self.end_effector_link_index,
                                                 position,
                                                 maxNumIterations=50,
                                                 jointDamping=n_joints*[0.005])
        arm_jpos = all_jpos[:len(self.arm_joint_ids)]
        return self.move_arm_to_jpos(arm_jpos)

    def move_arm_to_jpos(self, arm_jpos):
        for j_id, jpos in zip(self.arm_joint_ids, arm_jpos):
            pb.setJointMotorControl2(self._id,
                                     j_id,
                                     pb.POSITION_CONTROL,
                                     jpos,
                                     positionGain=0.1,
                                     maxVelocity=0.8)

        return self.monitor_movement(arm_jpos, self.arm_joint_ids)

    def set_gripper_state(self, gripper_state: float) -> bool:
        '''
        '''
        assert 0 <= gripper_state <= 1, 'Gripper state must be in range [0,1]'

        gripper_jpos = (1-gripper_state)*self.gripper_joint_limits[0] \
                       + gripper_state*self.gripper_joint_limits[1]

        pb.setJointMotorControlArray(self._id,
                                     self.gripper_joint_ids,
                                     pb.POSITION_CONTROL,
                                     gripper_jpos,
                                     positionGains=[0.2, 0.2])

        success = self.monitor_movement(gripper_jpos, self.gripper_joint_ids)
        return success

    def monitor_movement(self,
                         target_jpos: List[float],
                         joint_ids: List[int],
                        ) -> bool:
        old_jpos = list(zip(*pb.getJointStates(self._id, joint_ids)))[0]
        while True:
            [pb.stepSimulation() for _ in range(10)]

            time.sleep(0.01)

            achieved_jpos = list(zip(*pb.getJointStates(self._id, joint_ids)))[0]
            # print(list(zip(*pb.getJointStates(self._id, joint_ids)))[1])
            if np.allclose(target_jpos, achieved_jpos, atol=5e-3):
                # success
                return True

            if np.allclose(achieved_jpos, old_jpos, atol=1e-2):
                # movement stopped
                return False
            old_jpos = achieved_jpos


class SodaCan:
    def __init__(self):
        '''Mini coke can with texture'''
        self.height = 0.084
        self.radius = 0.0234
        self._id = pb.loadURDF('assets/urdf/soda_can.urdf',
                               (0,0,self.height/2),
                               (0,0,0,1))

        tex_id = pb.loadTexture('assets/textures/soda_can_tex.jpg')
        pb.changeVisualShape(self._id, -1,
                             textureUniqueId=tex_id)

    def get_pos(self):
        '''Get base position'''
        return pb.getBasePositionAndOrientation(self._id)[0]

    def set_pos(self, pos: List[float]):
        '''Reset base position'''
        pb.resetBasePositionAndOrientation(self._id, pos, (0,0,0,1))


def watch_ik_solution(robot, init_arm_jpos):
    target_pos = (0.12, 0.05, 0.06)

    axis_id = pb.loadURDF('assets/urdf/axis.urdf', target_pos, (0,0,0,1),
                          globalScaling=0.01, useFixedBase=True)

    while 1:
        # bad initial config
        robot.move_arm_to_jpos(init_arm_jpos)
        time.sleep(0.5)

        # solve ik and move to target
        robot.move_gripper_to(target_pos)

        # print position error
        ee_pos = pb.getLinkState(robot._id, robot.end_effector_link_index)[0]
        pos_err = np.linalg.norm(np.subtract(ee_pos, target_pos))
        print(f'Position error: {1000*pos_err:2f} mm')
        time.sleep(0.5)


if __name__ == "__main__":
    client = pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.loadURDF('plane.urdf')
    pb.setGravity(0,0,-10)

    pb.resetDebugVisualizerCamera(1, 50, -32, (-0.19, 0.16, -0.17))
    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
    # pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
    # pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
    # pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW,0)

    robot = RobotArm()

    bad_init_arm_jpos = [0,0,-0.1,0,0]
    watch_ik_solution(robot, bad_init_arm_jpos)

    good_init_arm_jpos = [0,-0.1,0.1,0,0]
    watch_ik_solution(robot, good_init_arm_jpos)

    can = SodaCan()
    can.set_pos((0.14, -0.01, can.height/2))

    robot.move_arm_to_jpos(good_init_arm_jpos)
    time.sleep(0.5)
    robot.set_gripper_state(robot.GRIPPER_OPENED)
    time.sleep(0.5)
    robot.move_gripper_to(can.get_pos())
    robot.set_gripper_state(robot.GRIPPER_CLOSED)
    time.sleep(0.5)
    robot.move_arm_to_jpos(robot.home_arm_jpos)
    time.sleep(0.5)

    while 1:
        pb.stepSimulation()
        time.sleep(1./256)
        pass
