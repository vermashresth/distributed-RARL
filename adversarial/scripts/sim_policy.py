import argparse

import joblib
import tensorflow as tf
import gym

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from test import test_const_adv, test_rand_adv, test_learnt_adv, test_rand_step_adv, test_step_adv

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    args = parser.parse_args()

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    with tf.Session() as sess:
        data = joblib.load(args.file)
        pro_policy = data['pro_policy']
        args_pickle = data['args']
        env = normalize(GymEnv(args_pickle.env, 0))
        while True:
            print(test_const_adv(env, pro_policy, path_length=args_pickle.path_length, n_traj=5, render=False, speedup=10000))
            if not query_yes_no('Continue simulation?'):
                break
