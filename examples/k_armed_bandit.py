import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from abc import abstractmethod


class kArmedBandit:
    def __init__(self, n_arms: int, seed: int=0) -> None:
        '''k Armed Bandit that randomly samples payout distributions for each
        lever

        Parameters
        ----------
        n_arms
            number of arms (ie levers) that can be pulled
        seed, default=0
            integer number used to seed the sampling of payout distributions
        '''
        np.random.seed(seed)

        self.n_arms = n_arms
        self._mus = np.random.uniform(-1, 0.4, size=n_arms)
        self._sigmas = np.random.uniform(0.1, 0.5, size=n_arms)

    def pull(self, arm_id: int) -> float:
        '''Pull arm and receive payout/reward'''
        return np.random.normal(self._mus[arm_id], self._sigmas[arm_id])

    def plot_distributions(self, n_samples: int=1000) -> None:
        data = np.array([np.random.normal(self._mus, self._sigmas)
                            for _ in range(n_samples)])

        plt.figure()
        plt.violinplot(data, showextrema=False, showmeans=True)
        plt.axhline(0, linestyle='--', color='k', linewidth=1)
        plt.xlabel('arm')
        plt.ylabel('reward')
        plt.title('Arm Payout Distributions')
        plt.show()


class Gambler:
    def __init__(self, n_arms: int,
                 initial_value: int=0) -> None:
        self.Q = np.full(n_arms, initial_value, dtype=float)
        self.N = np.zeros(n_arms, dtype=int)

    def update_value(self, arm_id: int, reward: float):
        raise NotImplemented

    @abstractmethod
    def select_action(self) -> int:
        pass


class ExplorationGambler(Gambler):
    '''Gambler that always explores new actions'''
    def select_action(self) -> int:
        pass


class ExploitationGambler(Gambler):
    '''Gambler that always exploits best-valued action'''
    def select_action(self) -> int:
        pass


class EGreedyGambler(Gambler):
    '''Gambler that always exploits best-valued action'''
    def __init__(self,
                 epsilon: float,
                 n_arms: int,
                 initial_value: int=0,
                 ) -> None:
        super().__init__(n_arms, initial_value)
        self.epsilon = epsilon

    def select_action(self) -> int:
        pass


def watch_method(bandit: kArmedBandit,
                 gambler: Gambler,
                 n_steps: int=500,
                ):
    rewards = []
    actions = []
    values = []
    for step in range(n_steps):
        arm_id = gambler.select_action()
        reward = bandit.pull(arm_id)

        gambler.update_value(arm_id, reward)

        rewards.append(reward)
        actions.append(arm_id)
        values.append(gambler.Q.copy())

    f, axs = plt.subplots(1, 2, figsize=(7, 3))

    axs[0].plot(np.arange(bandit.n_arms), bandit._mus, 'k.', label='actual')
    est, = axs[0].plot(np.arange(bandit.n_arms), values[0], 'b.', alpha=0.6, label='estimate')
    selected, = axs[0].plot([0], [0], 'r.', alpha=0.8)
    axs[0].set_ylim(np.min(bandit._mus)-0.5, np.max(bandit._mus)+0.5)
    axs[0].legend(loc=(0.5, 1.1))
    axs[0].set_xlabel('arms')
    axs[0].set_ylabel('values')
    plt.tight_layout()

    rew, = axs[1].plot([], [], 'k-')
    axs[1].set_ylim(np.min(rewards)-0.5, np.max(rewards)+0.5)
    axs[1].set_xlim(0, n_steps)
    axs[1].set_ylabel('rewards')
    axs[1].set_xlabel('actions taken')

    def animate(i):
        est.set_ydata(values[i])
        selected.set_xdata([actions[i]])
        selected.set_ydata([values[i][actions[i]]])
        rew.set_data(np.arange(i+1), rewards[:i+1])
        return est,

    anim = FuncAnimation(f, animate, interval=20, frames=n_steps)
    plt.draw()
    plt.show()


if __name__ == "__main__":
    bandit = kArmedBandit(8)
    bandit.show_distributions()

    # gambler = ExplorationGambler(bandit.n_arms)
    # watch_method(bandit, gambler)

