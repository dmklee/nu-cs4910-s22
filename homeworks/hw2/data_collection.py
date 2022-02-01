from typing import Tuple, List, Optional, Dict, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import numpy as np
import h5py
from tqdm import tqdm

from grasping_env import TopDownGraspingEnv
from utils import clamp_rotation

def foo(n_grasps, success_only, img_size, seed,
        render=False, show_progress=False):
    '''Runs env to collect samples, for parallelization'''
    np.random.seed(seed)
    env = TopDownGraspingEnv(img_size=img_size, render=render)

    imgs = []
    actions = []
    px_actions = []
    labels = []

    if show_progress:
        pbar = tqdm(total=n_grasps)

    while len(labels) < n_grasps:
        env.reset_object_position()
        img = env.take_picture()

        x,y,th = env.sample_expert_grasp()

        # add slight noise to expert
        x += np.random.uniform(-0.01, 0.01)
        y += np.random.uniform(-0.01, 0.01)
        th += np.random.uniform(-0.35, 0.35)

        # clip action accordingly
        x = np.clip(x, *env.workspace[:,0])
        y = np.clip(y, *env.workspace[:,1])
        th = clamp_rotation(th)

        success = env.perform_grasp(x, y, th)
        if success_only and not success:
            continue

        imgs.append(img)
        actions.append((x,y,th))
        labels.append(success)

        if show_progress:
            pbar.update(1)

    return imgs, actions, labels

def collect_dataset(dataset_name: str,
                    size: int,
                    success_only: bool=False,
                    img_size: int=42,
                    render: bool=True,
                    show_progress: bool=True,
                    seed: int=0,
                    n_processes: int=1,
                   ) -> None:
    '''Use this to watch data collection, as per Section 6. You are not graded
    on this, but it may be a helpful sanity check that your grasp success
    determination is accurate.

    Note
    ----
    The robot may occasionally go to some random configuration in the air, this
    is due to a bad IK solution.  There are ways to avoid this, but for now you
    can ignore it.
    '''
    with ProcessPoolExecutor() as executor:
        bg_futures = []
        for i in range(1, n_processes):
            new_future = executor.submit(foo, size//n_processes, success_only,
                                         img_size, seed=seed+1000*i)
            bg_futures.append(new_future)

        size_left = size - (n_processes-1) * (size//n_processes)
        imgs, actions, labels = foo(size_left, success_only, img_size, seed,
                                    render, show_progress)

        for f in as_completed(bg_futures):
            _imgs, _actions, _labels = f.result()
            imgs.extend(_imgs)
            actions.extend(_actions)
            labels.extend(_labels)

    # save it
    with h5py.File(dataset_name, 'w') as hf:
        hf.create_dataset('imgs', data=np.array(imgs))
        hf.create_dataset('actions', data=np.array(actions))
        hf.create_dataset('labels', data=np.array(labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, required=True,
                        help='File path where data will be saved')
    parser.add_argument('--size', '-s', type=int, default=10000,
                        help='Number of grasps in dataset')
    parser.add_argument('--success-only', action='store_true',
                        help='If true, only successful grasps will be included')
    parser.add_argument('--render', action='store_true',
                        help='If true, render gui during dataset collection')
    parser.add_argument('--seed', type=int, default=0,
                        help='Numpy random seed')
    parser.add_argument('--n-processes', '-np', type=int, default=4,
                        help='Number of parallel processes')
    args = parser.parse_args()

    collect_dataset(dataset_name= args.name,
                    size= args.size,
                    success_only= args.success_only,
                    render= args.render,
                    seed= args.seed,
                    n_processes= args.n_processes,
                   )
