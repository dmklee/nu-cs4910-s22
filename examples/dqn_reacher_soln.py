from typing import Tuple, Dict, Optional, List, Callable
import time
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import gym
import matplotlib.pyplot as plt
import pybullet as pb
from tqdm import tqdm as tqdm


class RobotArm:
    def __init__(self):
        '''xArm robot simulator used to move arm joints and perform forward
        kinematics
        '''
        self._id = pb.loadURDF("assets/urdf/xarm.urdf", basePosition=(0, 0, 0.012))

        # these are hard coded based on how urdf is written
        self.arm_joint_ids = [1,2,3,4,5]
        self.end_effector_link_index = 11

        # reset finger joints so gripper is closed
        hand_joint_ids = [6,7,8,9,10]
        hand_rest_states = [0.05, 0.05, 0.055, 0.0155, 0.031]
        [pb.resetJointState(self._id, j_id, jpos)
                 for j_id,jpos in zip(hand_joint_ids, hand_rest_states)]

    def teleport_arm(self, arm_jpos: List[float]) -> None:
        '''Changes joint states of arm; does not step simulator to keep it fast

        Parameters
        ----------
        arm_jpos
            joint positions in the same order as `self.arm_joint_ids`
        '''
        [pb.resetJointState(self._id, i, jp)
            for i,jp in zip(self.arm_joint_ids, arm_jpos)]

    def get_ee_pos(self) -> np.ndarray:
        '''Returns end effector position of arm'''
        return np.array(pb.getLinkState(self._id, self.end_effector_link_index)[4])


class ReacherEnv(gym.Env):
    def __init__(self,
                 episode_length: int=50,
                 action_step: float=0.035,
                 reward_type: str='sparse',
                 render: bool=False,
                ) -> None:
        '''Environment that implements 4-DoF Reacher arm.  State space is array
        of 4 joint positions. Actions are delta joint positions for single joint
        at a time, since the joint position can be changed in positive or negative
        direction, this results in 2*num_joints actions.  To reduce problem
        difficulty, the joint limits of the arm are restricted and the agent
        resets at a random joint state within these limits

        Parameters
        ----------
        episode_length
            number of actions that agent can take before environment is reset
        action_step
            delta joint position that is applied to a single joint while
            performing an action. If you make this too small, then the agent
            wont be able to reach the goal within the episode length; if you
            make this too large, then the agent wont have enough precision to
            achieve goal position
        reward_type : {'sparse', 'dense'}
            if sparse then agent receives +1 if within goal tolerance, 0 ow.
            if dense then agent receives inverse distance to goal at every
            time step
        render
            if true, then pybullet is initialized in GUI mode, otherwise use
            DIRECT mode.
        '''
        self.is_rendered = render
        pb.connect(pb.GUI if render else pb.DIRECT)
        if render:
            pb.resetDebugVisualizerCamera(0.8, 50, -32, (-0.19, 0.16, 0.))
            pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)

        self.robot = RobotArm()
        self.num_joints = 4
        self.joint_limits = np.array(((-0.5, -0.7, 0.5, 0.7),
                                      ( 0.5,  0.3, 1.5, 1.7)))

        assert reward_type in ('sparse', 'dense')
        self.reward_type = reward_type
        self.goal_ee_pos = np.array((0.18, 0.0, 0.2))
        self.goal_tol = 5e-3 # 1cm ball
        pb.createMultiBody(0, -1,
                           pb.createVisualShape(pb.GEOM_SPHERE,
                                                radius=self.goal_tol,
                                                rgbaColor=(0,1,0,0.4)),
                           self.goal_ee_pos)

        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                shape=(self.num_joints,))

        self.action_step = action_step
        self.action_space = gym.spaces.Discrete(2*self.num_joints)

        self.episode_length = episode_length

    def reset(self) -> np.ndarray:
        '''Randomly resets joint state of arm within joint limits and sets
        time step counter to 0

        Returns
        -------
        observation of joint positions
        '''
        # assign random start state
        self.joint_state = np.random.uniform(*self.joint_limits)
        self.robot.teleport_arm(self.joint_state)
        self.t_step = 0

        return self.get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        '''Performs action by changing single joint position by
        ±`self.action_step` while ensuring joint positions stay within limits

        Returns
        -------
        tuple of (obs, reward, done, info)
        '''
        assert self.action_space.contains(action)
        self.t_step += 1

        sign = 1 if action < self.num_joints else -1
        joint_id = action % self.num_joints

        self.joint_state[joint_id] += sign * self.action_step
        self.joint_state = np.clip(self.joint_state, *self.joint_limits)
        self.robot.teleport_arm(self.joint_state)

        obs = self.get_obs()
        reward = self.get_reward()
        done = self.is_done()
        info = {'success' : np.allclose(self.ee_pos, self.goal_ee_pos, atol=self.goal_tol)}

        # for rendering
        self.last_reward = reward

        return obs, reward, done, info

    def get_obs(self) -> np.ndarray:
        '''Returns observation which is a copy of current joint positions
        '''
        return self.joint_state.copy()

    def get_reward(self) -> float:
        '''Calculates reward based on distance from current end-effector position
        to goal end-effector position
        '''
        if self.reward_type == 'sparse':
            reward = np.allclose(self.ee_pos, self.goal_ee_pos, atol=self.goal_tol)
        elif self.reward_type == 'dense':
            dist_to_goal = np.linalg.norm(self.ee_pos-self.goal_ee_pos)
            reward = self.goal_tol/max(dist_to_goal, self.goal_tol)
            # raise NotImplemented

        return reward

    def is_done(self) -> bool:
        '''Environment should be reset if episode is over or agent has reached
        goal end effector position within a tolerance `self.goal_tol`
        '''
        return self.t_step >= self.episode_length \
                or np.allclose(self.ee_pos, self.goal_ee_pos, atol=self.goal_tol)

    @property
    def ee_pos(self):
        return self.robot.get_ee_pos()

    def render(self, sleep_time: float=0.1) -> None:
        if self.is_rendered:
            pb.addUserDebugLine(self.ee_pos,
                                self.goal_ee_pos,
                                (0,0,1), 1, 0.95*sleep_time)
            pb.addUserDebugText(f'R={self.last_reward:.2f}',
                                np.mean([self.ee_pos, self.goal_ee_pos], axis=0),
                                (0,0,1), 1, 0.95*sleep_time)
            time.sleep(sleep_time)


def watch_policy(env: ReacherEnv, policy: Optional[Callable]=None):
    '''Rolls out policy in environment and renders in GUI.  If policy is not
    provided then random policy is used

    Note
    ----
    environment must be initialized with `render=True`
    '''
    if policy is None:
        policy = lambda s: env.action_space.sample()

    s = env.reset()
    while 1:
        a = policy(s)
        sp, r, done, _ = env.step(a)
        env.render(0.1)

        s = sp
        if done:
            time.sleep(2.)
            s = env.reset()


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int) -> None:
        '''Q-Network instantiated as 3-layer MLP with 64 units

        Parameters
        ----------
        state_dim
            length of state vector
        n_actions
            number of actions in action space
        '''
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, n_actions)
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, s: Tensor) -> Tensor:
        '''Perform forward pass

        Parameters
        ----------
        s
            state tensor; shape=(B,|S|); dtype=float32
        Returns
        -------
        tensor of q values for each action; shape=(B,|A|); dtype=float32
        '''
        return self.layers(s)

    @torch.no_grad()
    def predict(self, s: Tensor) -> Tensor:
        '''Computes argmax over q-function at given states
        '''
        q_vals = self.forward(s)
        return torch.max(q_vals, dim=1)[1]

    def compute_loss(self, q_pred: Tensor, q_target: Tensor) -> Tensor:
        return self.loss_fn(q_pred, q_target)


class ReplayBuffer:
    def __init__(self, size: int, state_dim: int) -> None:
        self.data = {'s' : np.zeros((size, state_dim), dtype=np.float32),
                     'a' : np.zeros((size), dtype=np.int32),
                     'r' : np.zeros((size), dtype=np.float32),
                     'sp' : np.zeros((size, state_dim), dtype=np.float32),
                     'd' : np.zeros((size), dtype=np.bool8),
                    }

        self.size = size
        self.length = 0
        self._idx = 0

    def add_transition(self, s: np.ndarray, a: int, r: float,
                       sp: np.ndarray, d: bool) -> None:
        self.data['s'][self._idx] = s
        self.data['a'][self._idx] = a
        self.data['r'][self._idx] = r
        self.data['sp'][self._idx] = sp
        self.data['d'][self._idx] = d

        self._idx = (self._idx + 1) % self.size
        self.length = min(self.length + 1, self.size)

    def sample(self, batch_size: int) -> Tuple:
        idxs = np.random.randint(0, self.length, batch_size)

        s = self.data['s'][idxs]
        a = self.data['a'][idxs]
        r = self.data['r'][idxs]
        sp = self.data['sp'][idxs]
        d = self.data['d'][idxs]

        return s, a, r, sp, d


class Agent:
    def __init__(self,
                 env: ReacherEnv,
                 gamma: float=1.,
                 learning_rate: float=5e-4,
                 buffer_size: int=50000,
                 batch_size: int=128,
                 initial_epsilon: float=1.,
                 final_epsilon: float=0.01,
                 exploration_fraction: float=0.9,
                 target_network_update_freq: int=1000,
                 seed: int=0,
                 device: str='cpu',
                ) -> None:
        '''Agent that learns policy using DQN algorithm
        '''
        self.env = env

        assert 0 < gamma <= 1., 'Discount factor (gamma) must be in range (0,1]'
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.exploration_fraction = exploration_fraction

        self.buffer = ReplayBuffer(buffer_size, self.env.observation_space.shape[0])

        self.device = device
        self.network = QNetwork(self.env.observation_space.shape[0],
                                self.env.action_space.n).to(device)
        self.target_network = QNetwork(self.env.observation_space.shape[0],
                                       self.env.action_space.n).to(device)
        self.hard_target_update()

        self.optim = torch.optim.Adam(self.network.parameters(),
                                      lr= learning_rate)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)

    def train(self, num_steps: int) -> None:
        '''Trains q-network for given number of environment steps, plots
        rewards and loss curve
        '''
        rewards_data = []
        success_data = []
        loss_data = []

        episode_count = 0
        episode_rewards = 0
        opt_count = 0
        s = self.env.reset()

        pbar = tqdm(range(1, num_steps+1))
        for step in pbar:
            epsilon = self.compute_epsilon(step/(self.exploration_fraction*num_steps))
            a = self.select_action(s, epsilon)

            sp, r, done, info = self.env.step(a)
            episode_rewards += r

            self.buffer.add_transition(s=s, a=a, r=r, sp=sp, d=done)

            # optimize
            if self.buffer.length > self.batch_size:
                loss = self.optimize()
                opt_count += 1
                loss_data.append(loss)
                if opt_count % self.target_network_update_freq == 0:
                    self.hard_target_update()

            s = sp.copy()
            if done:
                s = self.env.reset()
                rewards_data.append(episode_rewards)
                success_data.append(info['success'])
                episode_rewards = 0
                episode_count += 1
                avg_success = np.mean(success_data[-min(episode_count, 50):])
                pbar.set_description(f'Success = {avg_success:.1%}')

        f, axs = plt.subplots(1,3, figsize=(7.5,2))
        axs[0].plot(np.convolve(rewards_data, np.ones(50)/50, 'valid'))
        axs[0].set_xlabel('episodes')
        axs[0].set_ylabel('sum of rewards')
        axs[1].plot(np.convolve(success_data, np.ones(50)/50, 'valid'))
        axs[1].set_xlabel('episodes')
        axs[1].set_ylabel('success rate')
        axs[2].plot(np.convolve(loss_data, np.ones(200)/200, 'valid'))
        axs[2].set_xlabel('opt steps')
        axs[2].set_ylabel('td loss')
        plt.tight_layout()
        plt.show()

    def optimize(self) -> float:
        '''Optimize Q-network by minimizing td-error on mini-batch sampled
        from replay buffer
        '''
        s,a,r,sp,d = self.buffer.sample(self.batch_size)

        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        a = torch.tensor(a, dtype=torch.long).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        sp = torch.tensor(sp, dtype=torch.float32).to(self.device)
        d = torch.tensor(d, dtype=torch.float32).to(self.device)

        q_pred = self.network(s).gather(1, a.unsqueeze(1)).squeeze()

        with torch.no_grad():
            q_target = r + self.gamma * torch.max(self.target_network(sp), dim=1)[0]

        self.optim.zero_grad()

        assert q_pred.shape == q_target.shape
        loss = self.network.compute_loss(q_pred, q_target)
        loss.backward()

        # it is common to clip gradient to prevent instability
        nn.utils.clip_grad_norm_(self.network.parameters(), 10)
        self.optim.step()
        return loss.item()

    def select_action(self, state: np.ndarray, epsilon: float=0.) -> int:
        '''Performs e-greedy action selection'''
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return self.policy(state)

    def compute_epsilon(self, fraction: float) -> float:
        '''Compute epsilon value based on fraction of training steps'''
        fraction = np.clip(fraction, 0., 1.)
        return (1-fraction) * self.initial_epsilon + fraction * self.final_epsilon

    def hard_target_update(self):
        '''Copy weights of q-network to target q-network'''
        self.target_network.load_state_dict(self.network.state_dict())

    def policy(self, state: np.ndarray) -> int:
        '''Calculates argmax of Q-function at given state'''
        t_state = torch.tensor(state, dtype=torch.float32,
                               device=self.device).unsqueeze(0)
        return self.network.predict(t_state).item()

if __name__ == "__main__":
    env = ReacherEnv(reward_type='dense', render=False)

    agent = Agent(env,
                  gamma=0.98,
                  learning_rate=1e-3,
                  buffer_size=10000,
                  initial_epsilon=0.1,
                  final_epsilon=0.01,
                  exploration_fraction=0.9,
                  target_network_update_freq=1000,
                  batch_size=256,
                  device='cpu',
                 )
    agent.train(20000)

    pb.disconnect()
    env = ReacherEnv(reward_type='dense', render=True)
    watch_policy(env, agent.policy)

