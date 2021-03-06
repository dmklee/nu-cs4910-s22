from typing import Tuple, List, Optional, Dict, Union
import argparse
import torch
import torch.nn as nn
import numpy as np
import pybullet as pb
import pybullet_data
from torchvision.transforms import ToTensor

from utils import WORKSPACE, IMG_SIZE

class RobotArm:
    GRIPPER_CLOSED = 0.
    GRIPPER_OPENED = 1.
    def __init__(self, fast_mode: bool=True):
        '''Robot Arm simulated in Pybullet, with support for performing top-down
        grasps within a specified workspace.
        '''
        # placing robot higher above ground improves top-down grasping ability
        self._id = pb.loadURDF("assets/urdf/xarm.urdf",
                               basePosition=(0, 0, 0.05),
                               # flags=pb.URDF_USE_SELF_COLLISION,
                              )

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

    def move_gripper_to(self, position: List[float], theta: float, teleport: bool=False):
        '''Commands motors to move end effector to desired position, oriented
        downwards with a rotation of theta about z-axis

        Parameters
        ----------
        position
            xyz position that end effector should move toward
        theta
            rotation (in radians) of the gripper about the z-axis.

        Returns
        -------
        bool
            True if movement is successful, False otherwise.
        '''
        quat = pb.getQuaternionFromEuler((0,-np.pi,theta))

        arm_jpos = self.solve_ik(position, quat)

        if teleport:
            self.teleport_arm(arm_jpos)
            return True
        else:
            return self.move_arm_to_jpos(arm_jpos)

    def solve_ik(self,
                 pos: List[float],
                 quat: Optional[List[float]]=None,
                ) -> Tuple[List[float], Dict[str, float]]:
        '''Calculates inverse kinematics solution for a desired end effector
        position and (optionally) orientation, and returns residuals

        Hint
        ----
        To calculate residuals, you can get the pose of the end effector link using
        `pybullet.getLinkState` (but you need to set the arm joint positions first)

        Parameters
        ----------
        pos
            target xyz position of end effector
        quat
            target orientation of end effector as unit quaternion if specified.
            otherwise, ik solution ignores final orientation

        Returns
        -------
        list
            joint positions of arm that would result in desired end effector
            position and orientation. in order from base to wrist
        dict
            position and orientation residuals:
                {'position' : || pos - achieved_pos ||,
                 'orientation' : 1 - |<quat, achieved_quat>|}
        '''
        old_arm_jpos = list(zip(*pb.getJointStates(self._id, self.arm_joint_ids)))[0]

        # good initial arm jpos for ik
        [pb.resetJointState(self._id, i, jp)
            for i,jp in zip(self.arm_joint_ids, self.home_arm_jpos)]

        n_joints = pb.getNumJoints(self._id)
        all_jpos = pb.calculateInverseKinematics(self._id,
                                                 self.end_effector_link_index,
                                                 pos,
                                                 quat,
                                                 maxNumIterations=20,
                                                 jointDamping=n_joints*[0.005])
        arm_jpos = all_jpos[:len(self.arm_joint_ids)]

        self.teleport_arm(old_arm_jpos)

        return arm_jpos

    def move_arm_to_jpos(self, arm_jpos: List[float]) -> bool:
        '''Commands motors to move arm to desired joint positions

        Parameters
        ----------
        arm_jpos
            joint positions (radians) of arm joints, ordered from base to wrist

        Returns
        -------
        bool
            True if movement is successful, False otherwise.
        '''
        # cannot use setJointMotorControlArray because API does not expose
        # maxVelocity argument, which is needed for stable object manipulation
        for j_id, jpos in zip(self.arm_joint_ids, arm_jpos):
            pb.setJointMotorControl2(self._id,
                                     j_id,
                                     pb.POSITION_CONTROL,
                                     jpos,
                                     positionGain=0.2,
                                     maxVelocity=0.8)

        return self.monitor_movement(arm_jpos, self.arm_joint_ids)

    def teleport_arm(self, arm_jpos: List[float]) -> None:
        [pb.resetJointState(self._id, i, jp)
            for i,jp in zip(self.arm_joint_ids, arm_jpos)]

    def teleport_gripper(self, gripper_state: float) -> None:
        assert 0 <= gripper_state <= 1, 'Gripper state must be in range [0,1]'

        gripper_jpos = (1-gripper_state)*self.gripper_joint_limits[0] \
                       + gripper_state*self.gripper_joint_limits[1]
        [pb.resetJointState(self._id, i, jp)
            for i,jp in zip(self.gripper_joint_ids, gripper_jpos)]

    def set_gripper_state(self, gripper_state: float) -> bool:
        '''Commands motors to move gripper to given state

        Parameters
        ----------
        gripper_state
            gripper state is a continuous number from 0. (fully closed)
            to 1. (fully open)

        Returns
        -------
        bool
            True if movement is successful, False otherwise.

        Raises
        ------
        AssertionError
            If `gripper_state` is outside the range [0,1]
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
        '''Monitors movement of motors to detect early stoppage or success.

        Note
        ----
        Current implementation calls `pybullet.stepSimulation`, without which the
        simulator will not move the motors.  You can avoid this by setting
        `pybullet.setRealTimeSimulation(True)` but this is usually not advised.

        Parameters
        ----------
        target_jpos
            final joint positions that motors are moving toward
        joint_ids
            the joint ids associated with each `target_jpos`, used to read out
            the joint state during movement

        Returns
        -------
        bool
            True if movement is successful, False otherwise.
        '''
        old_jpos = list(zip(*pb.getJointStates(self._id, joint_ids)))[0]
        while True:
            [pb.stepSimulation() for _ in range(10)]

            achieved_jpos = list(zip(*pb.getJointStates(self._id, joint_ids)))[0]
            if np.allclose(target_jpos, achieved_jpos, atol=1e-3):
                # success
                return True

            if np.allclose(achieved_jpos, old_jpos, atol=1e-2):
                # movement stopped
                return False
            old_jpos = achieved_jpos


class Camera:
    def __init__(self, workspace: np.ndarray, img_size: int) -> None:
        '''Camera that is mounted to view workspace from above

        Hint
        ----
        For this camera setup, it may be easiest if you use the functions
        `pybullet.computeViewMatrix` and `pybullet.computeProjectionMatrixFOV`.
        cameraUpVector should be (0,1,0)

        Parameters
        ----------
        workspace
            2d array describing extents of robot workspace that is to be viewed,
            in the format: ((min_x,min_y), (max_x, max_y))

        Attributes
        ----------
        img_size : int
            height, width of rendered image
        view_mtx : List[float]
            view matrix that is positioned to view center of workspace from above
        proj_mtx : List[float]
            proj matrix that set up to fully view workspace
        '''
        self.img_size = img_size

        cam_height = 0.25
        workspace_width = workspace[1,0] - workspace[0,0]
        fov = 2 * np.degrees(np.arctan2(workspace_width/2, cam_height))

        cx, cy = np.mean(workspace, axis=0)
        eye_pos = (cx, cy, cam_height)
        target_pos = (cx, cy, 0)
        self.view_mtx = pb.computeViewMatrix(cameraEyePosition=eye_pos,
                                             cameraTargetPosition=target_pos,
                                            cameraUpVector=(-1,0,0))
        self.proj_mtx = pb.computeProjectionMatrixFOV(fov=fov,
                                                      aspect=1,
                                                      nearVal=0.01,
                                                      farVal=1)

    def get_rgb_image(self) -> np.ndarray:
        '''Takes rgb image

        Returns
        -------
        np.ndarray
            shape (H,W,3) with dtype=np.uint8
        '''
        rgba = pb.getCameraImage(width=self.img_size,
                                 height=self.img_size,
                                 viewMatrix=self.view_mtx,
                                 projectionMatrix=self.proj_mtx,
                                 renderer=pb.ER_TINY_RENDERER)[2]

        return rgba[...,:3]


class TopDownGraspingEnv:
    def __init__(self, img_size: int=IMG_SIZE, render: bool=True) -> None:
        '''Pybullet simulator with robot that performs top down grasps of a
        single object.  A camera is positioned to take images of workspace
        from above.
        '''
        self.client = pb.connect(pb.GUI if render else pb.DIRECT)
        pb.setPhysicsEngineParameter(numSubSteps=0,
                                     numSolverIterations=100,
                                     solverResidualThreshold=1e-7,
                                     constraintSolverType=pb.CONSTRAINT_SOLVER_LCP_SI)
        pb.setGravity(0,0,-10)

        # create ground plane
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        # offset plane y-dim to place white tile under workspace
        self.plane_id = pb.loadURDF('plane.urdf', (0,-0.5,0))

        # makes collisions with plane more stable
        pb.changeDynamics(self.plane_id, -1,
                          linearDamping=0.04,
                          angularDamping=0.04,
                          restitution=0,
                          contactStiffness=3000,
                          contactDamping=100)

        # add robot
        self.robot = RobotArm()

        # add object
        self.object_id = pb.loadURDF("assets/urdf/object.urdf")
        pb.changeDynamics(self.object_id, -1,
                          lateralFriction=1,
                          spinningFriction=0.005,
                          rollingFriction=0.005)
        self.object_width = 0.02

        self.workspace = WORKSPACE

        self.grasp_height = self.object_width/2
        if render:
            self.draw_workspace()

        # add camera
        self.camera = Camera(self.workspace, img_size)

    def draw_workspace(self) -> None:
        '''This is just for visualization purposes, to help you with the object
        resetting.  Must be in GUI mode, otherwise error occurs

        Note
        ----
        Pybullet debug lines only show up in GUI mode so they won't help you
        with camera placement.
        '''
        corner_ids = ((0,0), (0,1), (1,1), (1,0), (0,0))
        for i in range(4):
            start = (*self.workspace[corner_ids[i],[0,1]], 0.)
            end = (*self.workspace[corner_ids[i+1],[0,1]], 0.)
            pb.addUserDebugLine(start, end, (0,0,0), 3)

    def perform_grasp(self, x, y, theta) -> bool:
        '''Perform top down grasp in the workspace.  All grasps will occur
        at a height of the center of mass of the object (i.e. object_width/2)

        Parameters
        ----------
        x
            x position of the grasp in world frame
        y
            y position of the grasp in world frame
        theta
            target rotation about z-axis of gripper during grasp

        Returns
        -------
        bool
            True if object was successfully grasped, False otherwise. It is up
            to you to decide how to determine success
        '''
        self.robot.move_arm_to_jpos(self.robot.home_arm_jpos)
        self.robot.teleport_gripper(self.robot.GRIPPER_OPENED)

        pos = np.array((x, y, self.grasp_height))

        self.robot.move_gripper_to(pos, theta, teleport=True)
        self.robot.set_gripper_state(self.robot.GRIPPER_CLOSED)

        self.robot.move_arm_to_jpos(self.robot.home_arm_jpos)

        # check if object is above plane
        min_object_height = 0.2
        obj_height = pb.getBasePositionAndOrientation(self.object_id)[0][2]
        success = obj_height > min_object_height

        return success

    def sample_random_grasp(self) -> Tuple[float, float, float]:
        '''Samples random grasp (x,y,theta) located within the workspace

        Returns
        -------
        tuple
            x, y, theta (radians)
        '''
        x,y = np.random.uniform(*self.workspace)
        theta = np.random.uniform(0, 2*np.pi)
        return x, y, theta

    def sample_expert_grasp(self) -> Tuple[float, float, float]:
        '''Samples expert grasp (x,y,theta) located within the workspace. An
        expert grasp can be determined using information about the state of the
        object. It should have >90% success rate if implemented correctly.

        Returns
        -------
        tuple
            x, y, theta (radians)
        '''
        obj_pos, obj_quat = pb.getBasePositionAndOrientation(self.object_id)
        x, y = obj_pos[:2]
        theta = pb.getEulerFromQuaternion(obj_quat)[2]

        x,y = np.clip((x,y), self.workspace[0], self.workspace[1])

        return x, y, theta

    def reset_object_position(self, full_rotation: bool=True) -> None:
        '''Places object randomly in workspace.  The x,y position should be
        within the workspace, and the rotation performed only about z-axis.
        The height of the object should be set such that it sits on the plane
        '''
        ws_padding = 0.01
        x,y = np.random.uniform(self.workspace[0]+ws_padding,
                                self.workspace[1]-ws_padding)
        theta = np.random.uniform(0, 2*np.pi)

        pos = np.array((x,y,self.object_width/2))
        quat = pb.getQuaternionFromEuler((0,0, theta))
        pb.resetBasePositionAndOrientation(self.object_id, pos, quat)

    def take_picture(self) -> np.ndarray:
        '''Takes picture using camera

        Returns
        -------
        np.ndarray
            rgb image of shape (H,W,3) and dtype of np.uint8
        '''
        return self.camera.get_rgb_image()


def test_grasp_prediction(env: TopDownGraspingEnv, network: nn.Module):
    n_attempts = 0
    n_successes = 0
    while 1:
        env.reset_object_position()
        img = env.take_picture()

        t_img = ToTensor()(img).unsqueeze(0)
        with torch.no_grad():
            x,y,th = network(t_img).squeeze().numpy()

        success = env.perform_grasp(x, y, th)

        n_attempts += 1
        n_successes += int(success)
        message = 'SUCCESS' if success else 'FAILURE'
        print(f"{message} | Success rate={(n_successes/n_attempts):.1%} on {n_attempts} attempts")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', '-mp', type=str, required=True,
                        help='Path to model file')
    args = parser.parse_args()

    env = TopDownGraspingEnv(render=True)
    pb.resetDebugVisualizerCamera(1, 50, -32, (-0.19, 0.16, -0.17))
    # pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
    pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    # pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

    from networks import GraspPredictorNetwork
    model = GraspPredictorNetwork([3, IMG_SIZE, IMG_SIZE], 3)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    test_grasp_prediction(env, model)
