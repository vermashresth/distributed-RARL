from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.random_uniform_control_policy import RandomUniformControlPolicy
from rllab.policies.constant_control_policy import ConstantControlPolicy
from IPython import embed
import matplotlib.pyplot as plt
import rllab.misc.logger as logger
import numpy as np
from test import test_const_adv, test_rand_adv, test_learnt_adv, test_rand_step_adv, test_step_adv
import pickle
import argparse
import os
import gym
import joblib

## Pass arguments ##
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True, help='path of weight file')
parser.add_argument('--env', type=str, help='Name of adversarial environment')
parser.add_argument('--adv_name', type=str, default='no_adv', help='DOESNT MATTER FOR THIS BASELINE')
parser.add_argument('--path_length', type=int, default=1000, help='maximum episode length')
parser.add_argument('--layer_size', nargs='+', type=int, default=[100,100,100], help='layer definition')
parser.add_argument('--if_render', type=int, default=0, help='Should we animate at all?')
parser.add_argument('--after_render', type=int, default=100, help='After how many to animate')
parser.add_argument('--n_exps', type=int, default=1, help='')
parser.add_argument('--n_itr', type=int, default=100, help='')
parser.add_argument('--n_pro_itr', type=int, default=1, help='')
parser.add_argument('--n_adv_itr', type=int, default=1, help='')
parser.add_argument('--batch_size', type=int, default=4000, help='')
parser.add_argument('--filename_append', type=str, default='', help='stuff to append to save filename')
parser.add_argument('--save_every', type=int, default=100, help='')
parser.add_argument('--n_process', type=int, default=10, help='Number of threads for sampling environment')
parser.add_argument('--adv_fraction', type=float, default=1.0, help='DOESNT MATTER')
parser.add_argument('--step_size', type=float, default=0.01, help='step size for learner')
parser.add_argument('--gae_lambda', type=float, default=0.97, help='gae_lambda for learner')

args_init = parser.parse_args()

data = joblib.load(args_init.file)
pro_policy = data['pro_policy']
args = data['args']

## Number of experiments to run ##
env_name = args.env
adv_name = 'no_adv'
path_length = args.path_length
layer_size = tuple(args.layer_size)
ifRender = bool(args.if_render)
afterRender = args.after_render
n_exps = args.n_exps
n_itr = args.n_itr
n_pro_itr = args.n_pro_itr
n_adv_itr = args.n_adv_itr
batch_size = args.batch_size
save_every = args.save_every
n_process = args.n_process
adv_fraction = 1.0
step_size = args.step_size
gae_lambda = args.gae_lambda


#overwrite arguymenstsss
save_every = args_init.save_every
n_itr= args_init.n_itr
ifRender=bool(args_init.if_render)

const_test_rew_summary = data['zero_test']
rand_test_rew_summary =data ['rand_test']
step_test_rew_summary = data['step_test']
rand_step_test_rew_summary = data['rand_step_test']
adv_test_rew_summary = data['adv_test']
ne = data['exp_save']
ni = data['iter_save']

save_prefix = 'BASELINE-env-{}_{}_Exp{}_Itr{}_BS{}_Adv{}_stp{}_lam{}'.format(env_name, adv_name, n_exps, n_itr, batch_size, adv_fraction, step_size, gae_lambda)
save_dir = os.environ['HOME']+'/btpstuff/rllab-adv/results/baselines'
fig_dir = 'figs'
save_name = save_dir+'/'+save_prefix+'.p'
fig_name = fig_dir+'/'+save_prefix+'.png'

while ne < n_exps:
    ## Environment definition ##
    env = normalize(GymEnv(env_name, adv_fraction))
    ## Protagonist policy definition ##
    pro_policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=layer_size,
        is_protagonist=True
    )
    pro_baseline = LinearFeatureBaseline(env_spec=env.spec)

    ## Zero Adversary for the protagonist training ##
    zero_adv_policy = ConstantControlPolicy(
        env_spec=env.spec,
        is_protagonist=False,
        constant_val = 0.0
    )

    ## Optimizer for the Protagonist ##
    from rllab.sampler import parallel_sampler
    parallel_sampler.initialize(n_process)
    if adv_name=='no_adv':
        pro_algo = TRPO(
            env=env,
            pro_policy=pro_policy,
            adv_policy=zero_adv_policy,
            pro_baseline=pro_baseline,
            adv_baseline=pro_baseline,
            batch_size=batch_size,
            max_path_length=path_length,
            n_itr=n_pro_itr,
            discount=0.995,
            gae_lambda=gae_lambda,
            step_size=step_size,
            is_protagonist=True
        )

    ## Joint optimization ##
    if ifRender==True: test_const_adv(env, pro_policy, path_length=path_length, n_traj = 1, render=True)
    pro_rews = []
    adv_rews = []
    all_rews = []
    const_testing_rews = []
    const_testing_rews.append(test_const_adv(env, pro_policy, path_length=path_length))
    rand_testing_rews = []
    rand_testing_rews.append(test_rand_adv(env, pro_policy, path_length=path_length))
    step_testing_rews = []
    step_testing_rews.append(test_step_adv(env, pro_policy, path_length=path_length))
    rand_step_testing_rews = []
    rand_step_testing_rews.append(test_rand_step_adv(env, pro_policy, path_length=path_length))
    adv_testing_rews = []
    adv_testing_rews.append(test_rand_adv(env, pro_policy, path_length=path_length))
    #embed()
    while ni < n_itr:
        logger.log('\n\n\n####expNO{}_{} global itr# {}####\n\n\n'.format(ne,adv_name,ni))
        pro_algo.train()
        pro_rews += pro_algo.rews; all_rews += pro_algo.rews;
        logger.log('Protag Reward: {}'.format(np.array(pro_algo.rews).mean()))
        #const_testing_rews.append(test_const_adv(env, pro_policy, path_length=path_length))
        #rand_testing_rews.append(test_rand_adv(env, pro_policy, path_length=path_length))
        #step_testing_rews.append(test_step_adv(env, pro_policy, path_length=path_length))
        #rand_step_testing_rews.append(test_rand_step_adv(env, pro_policy, path_length=path_length))
        #adv_testing_rews.append(test_rand_adv(env, pro_policy, path_length=path_length))
        if ni!=0 and ni%save_every==0:
            ## SAVING INFO ##
            pickle.dump({'args': args,
                         'pro_policy': pro_policy,
                         'zero_test': const_test_rew_summary,
                         'rand_test': rand_test_rew_summary,
                         'step_test': step_test_rew_summary,
                         'rand_step_test': rand_step_test_rew_summary,
                         'iter_save': ni,
                         'exp_save': ne,
                         'adv_test': adv_test_rew_summary}, open(save_name+'.temp','wb'))
        ni = ni + 1

    ## Shutting down the optimizer ##
    pro_algo.shutdown_worker()
    const_test_rew_summary.append(const_testing_rews)
    rand_test_rew_summary.append(rand_testing_rews)
    step_test_rew_summary.append(step_testing_rews)
    rand_step_test_rew_summary.append(rand_step_testing_rews)
    adv_test_rew_summary.append(adv_testing_rews)
    ne = ne + 1

## SAVING INFO ##
pickle.dump({'args': args,
             'pro_policy': pro_policy,
             'zero_test': const_test_rew_summary,
             'rand_test': rand_test_rew_summary,
             'step_test': step_test_rew_summary,
             'rand_step_test': rand_step_test_rew_summary,
             'adv_test': adv_test_rew_summary}, open(save_name,'wb'))

logger.log('\n\n\n#### DONE ####\n\n\n')
