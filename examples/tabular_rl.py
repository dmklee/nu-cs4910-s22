from typing import Optional, Callable, Tuple, Dict
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import gym
from gym.envs.classic_control import rendering
# from gym.utils import pyglet_rendering as rendering

class GridWorldEnv(gym.Env):
    def __init__(self,
                 width: int=5,
                 max_episode_length: int=30,
                 seed: Optional[int]=42,
                ) -> None:
        '''2D grid world environment

        Parameters
        ----------
        width
            size of grid world in both dimensions. e.g. width=4 means there is
            a 4x4 grid (16 states total)
        max_episode_length
            number of actions that can be taken before environment is
            automatically reset

        Attributes
        ----------
        observation_space : gym.Space
            observations are integer values ranging from 0 to width^2-1
        action_space : gym.Space
            actions are integer values ranging from 0 to 3, indicating which
            of the four cardinal directions to move
        '''
        np.random.seed(seed)

        self.width = width
        self.max_episode_length = max_episode_length

        self.observation_space = gym.spaces.Discrete(width**2)

        self.action_space = gym.spaces.Discrete(4)
        self._actions = np.array(((1,0),(0,1),(-1,0),(0,-1)), dtype=int)

        self.reset_state = np.array((0,0), dtype=int)
        self.goal_state = np.array((width-2,width-2), dtype=int)
        self.avoid_state = None

    def reset(self) -> int:
        '''Resets the position of the agent in the environment.  Agent is always
        reset to the lower left corner of the environment.

        Returns
        -------
        observation of the environment
        '''
        self.state = self.reset_state.copy()
        self.t_step = 0

        return self.get_obs()

    def get_obs(self) -> int:
        '''Returns observation associated with current state.  Observation
        is the state unraveled to one dimension
        '''
        return self.state[0] + self.state[1]*self.width

    def get_reward(self) -> float:
        '''Gets reward associated with given state, should be called before
        setting new state during step
        '''
        return 1. * np.allclose(self.state, self.goal_state)

    def step(self, action: int) -> Tuple[int, bool, bool, Dict]:
        '''Performs action by moving agent unless agent tries to move outside
        of grid

        Raises
        ------
        AssertionError
            action argument is outside of the action space of the environment

        Parameters
        ----------
        action
            integer describing action direction

        Returns
        -------
        tuple containing observation, reward, done and info.  the environment
        is done if reward=1 or if number of actions taken >= max_episode_length
        '''
        assert self.action_space.contains(action)

        reward = self.get_reward()

        self.state = np.add(self.state, self._actions[action])
        self.state = np.clip(self.state, 0, self.width-1)

        self.t_step += 1

        obs = self.get_obs()
        done = reward != 0 or self.t_step >= self.max_episode_length
        info = {}

        return obs, reward, done, info

    def render(self,
               mode: str="human",
               values: Optional[np.ndarray]=None,
               actions: Optional[np.ndarray]=None,
              ) -> None:
        '''Renders environment using pyglet'''
        def index_to_location(i):
            x = (i % self.width) * grid_size + margin
            y = (i // self.width) * grid_size + margin
            return x, y

        ############################
        # initialize viewer geoms
        ############################
        screen_size = 400
        margin = 50
        grid_size = (screen_size - 2 * margin) / (self.width - 1)
        if "viewer" not in dir(self):
            self.viewer = rendering.Viewer(screen_size, screen_size)

            self.viewer_grid_cells = []
            self.viewer_policy_arrow_transforms = []
            # add background grid
            tmp = grid_size / 2
            vertices = ((-tmp, -tmp), (-tmp, tmp), (tmp, tmp), (tmp, -tmp))
            arrow = ((-grid_size/6, 0), (grid_size/4, 0),
                     (grid_size/8, grid_size/8), (grid_size/4, 0),
                     (grid_size/8, -grid_size/8), (grid_size/4, 0))

            for i in range(self.width ** 2):
                grid_cell = rendering.make_polygon(vertices, filled=True)
                grid_cell.set_color(1,1,1)
                grid_cell.add_attr(rendering.Transform(index_to_location(i)))
                self.viewer.add_geom(grid_cell)
                self.viewer_grid_cells.append(grid_cell)

                grid_border = rendering.make_polygon(vertices, filled=False)
                grid_border.set_color(0,0,0)
                grid_border.set_linewidth(3)
                grid_border.add_attr(rendering.Transform(index_to_location(i)))
                self.viewer.add_geom(grid_border)

                grid_arrow = rendering.make_polygon(arrow, filled=False)
                grid_arrow.set_color(0,0,0)
                grid_arrow.set_linewidth(4)
                grid_arrow_transform = rendering.Transform()
                grid_arrow.add_attr(grid_arrow_transform)
                self.viewer.add_geom(grid_arrow)
                self.viewer_policy_arrow_transforms.append(grid_arrow_transform)

            # add agent
            tmp = grid_size * 0.15
            vertices = ((-tmp, -tmp), (-tmp, tmp), (tmp, tmp), (tmp, -tmp))
            agent_geom = rendering.make_polygon(vertices)
            agent_geom.set_color(0.2, 0.3, 0.8)
            agent_loc = (
                self.state[0] * grid_size + margin,
                self.state[1] * grid_size + margin,
            )
            self.viewer_agent_transform = rendering.Transform(agent_loc)
            agent_geom.add_attr(self.viewer_agent_transform)
            self.viewer.add_geom(agent_geom)

            # add goal
            if self.avoid_state is not None:
                avoid_geom = rendering.make_circle(radius=0.15 * grid_size)
                avoid_geom.set_color(0.8, 0.2, 0.3)
                avoid_loc = (
                    self.avoid_state[0] * grid_size + margin,
                    self.avoid_state[1] * grid_size + margin,
                )
                self.viewer_avoid_transform = rendering.Transform(avoid_loc)
                avoid_geom.add_attr(self.viewer_avoid_transform)
                self.viewer.add_geom(avoid_geom)

            # add goal
            goal_geom = rendering.make_circle(radius=0.15 * grid_size)
            goal_geom.set_color(0.2, 0.8, 0.3)
            goal_loc = (
                self.goal_state[0] * grid_size + margin,
                self.goal_state[1] * grid_size + margin,
            )
            self.viewer_goal_transform = rendering.Transform(goal_loc)
            goal_geom.add_attr(self.viewer_goal_transform)
            self.viewer.add_geom(goal_geom)
        ############################

        # set actions
        if actions is not None:
            for i, a in enumerate(actions):
                self.viewer_policy_arrow_transforms[i].set_translation(
                    *index_to_location(i)
                )
                self.viewer_policy_arrow_transforms[i].set_rotation( a * np.pi/2 )
        else:
            [tfm.set_translation(-10,-10)
                for tfm in self.viewer_policy_arrow_transforms]

        # set values
        if values is not None:
            if np.max(values) > np.min(values):
                norm_values = (values - np.min(values)) / (np.max(values) - np.min(values))
                rgbs = cm.get_cmap('RdYlGn')(norm_values)[:,:3]
            else:
                rgbs = np.ones((len(values), 3))
            for rgb, gc in zip(rgbs, self.viewer_grid_cells):
                gc.set_color(*rgb)

        # move agent
        agent_loc = (
            self.state[0] * grid_size + margin,
            self.state[1] * grid_size + margin,
        )
        self.viewer_agent_transform.set_translation(*agent_loc)
        return self.viewer.render(return_rgb_array=(mode == "rgb_array"))


def watch_policy(env: gym.Env,
                 policy: Optional[Callable]=None,
                ) -> None:
    '''Renders environment while perfoming a policy
    '''
    if policy is None:
        policy = lambda s: env.action_space.sample()

    s = env.reset()
    while 1:
        env.render()
        time.sleep(0.1)
        a = policy(s)
        s, r, d, _ = env.step(a)
        if r:
            time.sleep(0.5)
        if d:
            s = env.reset()


def policy_evaluation(env: gym.Env,
                      policy: Callable,
                      alpha: float=0.1,
                      gamma: float=0.99,
                      num_steps: int=10000,
                      render_freq: int=20, # in terms of episodes
                     ) -> np.ndarray:
    '''Use TD-learning to learn a value function for a given policy
    '''
    V = None

    s = env.reset()
    episode_id = 0
    for t in range(num_steps):
        if render_freq and episode_id % render_freq == 0:
            env.render(values=V)
            time.sleep(0.05)

        a = policy(s)
        s_p, r, d, _ = env.step(a)

        #############################
        # perform TD-update
        #############################

        s = s_p
        if d:
            s = env.reset()
            episode_id += 1

def q_learning(env: gym.Env,
               alpha: float=0.1,
               gamma: float=0.9,
               num_steps: int=10000,
               epsilon: float=0.1,
               render_freq: int=50, # in terms of episodes
              ):
    '''Use q-learning to perform policy evaluation and improvement

    Parameters
    ----------
    env
        environment that is trained on
    alpha
        learning rate of update step
    gamma
        discount factor
    num_steps
        number of environment steps during training
    epsilon
        value used for e-greedy action selection
    render_freq
        how often should an episode be rendered, if 0 then never rendered
    '''
    Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float32)

    rewards_data = []
    episode_lengths = []

    s = env.reset()
    episode_id = 0
    for t in range(num_steps):
        if render_freq and episode_id % render_freq == 0:
            env.render(values=np.max(Q, axis=1), actions=np.argmax(Q, axis=1))
            time.sleep(0.05)

        #############################
        # select action
        #############################

        s_p, r, d, _ = env.step(a)
        #############################
        # perform Q-learning update
        #############################
        s = s_p

        if d:
            rewards_data.append(r)
            episode_lengths.append(env.t_step)
            s = env.reset()
            episode_id += 1

    env.render(values=np.max(Q, axis=1), actions=np.argmax(Q, axis=1))

    smooth_window = 50
    f, axs = plt.subplots(1,2)
    rewards_smoothed = np.convolve(rewards_data,
                                   np.ones(smooth_window)/smooth_window,
                                   mode='valid')
    axs[0].plot(rewards_smoothed)
    axs[0].set_xlabel('episodes')
    axs[0].set_ylabel('average rewards')

    lengths_smoothed = np.convolve(episode_lengths,
                                   np.ones(smooth_window)/smooth_window,
                                   mode='valid')
    axs[1].plot(lengths_smoothed)
    axs[1].axhline(2*(env.width-1)-1, color='k', linestyle=':')
    axs[1].set_xlabel('episodes')
    axs[1].set_ylabel('episode length')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    env = GridWorldEnv()
    watch_policy(env)

    # policy_evaluation(env, lambda s: env.action_space.sample())
    # q_learning(env)

