#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import argparse
import numpy as np

sys.path.append(os.path.join(os.pardir, 'src'))

from datetime import datetime

from learn_rl_ds import RL_DS
from utils.utils import mse, time_stamp
from utils.log_config import logger
from utils.plot_tools import plot_ds_stream
from learn_gmm_ds import expert_seds_model, prepare_expert_data

from utils.data_loader import load_custom_data
from utils.model_evaluation import generate_trajectories, compute_reference_mse

np.random.seed(0)


def train_rl_policy(data_file_path: str, learning_method: str = "BC", policy_agent: str = 'PPO',
    motion_shape: str = "G", n_dems: int = 50, n_epochs_bc: int = 100,
    n_timesteps_gail: int = int(3e5), warm_policy : bool = True, plot: bool = False,
    model_name: str = 'test', save: bool = False, save_dir: str = ""):
    """ Training sequence for a neural network or polynomial to estimate a nonlinear dynamical system.

    Args:
        learning_method(str, optional): Type of the policy learning method, could be either
            Behavioral Cloning ("BC"), or Generative Adversarial Imitation Learning ("GAIL").

        policy_agent(str, optional): Policy agent, can only be "PPO".
        motion_shape (str, optional): Shape of the trajectories. Defaults to "G".
        n_dems (int, optional): Total number of augmented demonstrations. Defaults to 50.
        plot (bool, optional): Whether to plot trajectories and final ds or not. Defaults to False.
        n_epochs_bc (int, optional): Total number of epochs for Behavioral Cloning.
            Defaults to 1000.

        warm_policy (bool, optional): Whether to pass a working policy to GAIL. Defaults to False.
        n_timesteps_gail(int, optional): Total timestemps to train with GAIL. Defaults to 3e5.
        model_name (str, optional): Name of the model for save and load. Defaults to 'test'.
    """
    name = f'{model_name}-{learning_method.lower()}-{data_file_path.split("/")[-1].split(".")[0]}-{time_stamp()}'

    ''' Load dataset '''
    aug_trajs, aug_vels, num_samples = load_custom_data(data_file_path=data_file_path, normalized=True)
    data_dim = aug_trajs.shape[1]

    ''' Train and save a model'''
    if data_dim == 2:
        rl_ds = RL_DS(learning_method, gym_env="taskspace2d", learner_agent=policy_agent,
                      gym_envs_path=os.path.join(os.pardir, 'src', 'envs'))
    else:
        rl_ds = RL_DS(learning_method, gym_env="taskspacexd", learner_agent=policy_agent,
                      gym_envs_path=os.path.join(os.pardir, 'src', 'envs'), data_dim=data_dim)
    rl_ds.fit(aug_trajs, aug_vels, n_epochs_bc, n_timesteps_gail, True, warm_policy, n_dems)

    ''' Test the model '''
    trajectories_py, velocities_py, num_samples = load_custom_data(data_file_path=data_file_path, normalized=True)
    preds = rl_ds.predict(trajectories_py)
    err = mse(preds, velocities_py)
    logger.info(f'Final MSE on data: {err:.4f}')

    ''' Plot the DS '''
    if plot:
        plot_ds_stream(rl_ds, trajectories_py, save_dir=save_dir, file_name=f'ds-{name}',
                       show_legends=False)

    ''' Save the DS '''
    if save:
        print(f'[INFO] Saving model in {save_dir}/{name}/')
        rl_ds.save(model_name=name, dir=save_dir)

    # Generate evaluation trajectories by incrementally running model prediction on initial states
    trajectories_file_name = f'{name}-evaluation-trajectories-{datetime.now().strftime("%d-%m-%H-%M")}'
    generate_trajectories(rl_ds, trajectories_py,  n_samples=num_samples, save_dir=save_dir, file_name=trajectories_file_name)

    print(f'[INFO] MSE between reference and predicted velocities at all reference positions: {err:.10f}')

    return err, rl_ds

if __name__ == '__main__':
    # argument parser initiation
    parser = argparse.ArgumentParser(description='Nonlinear DS experiments CLI interface.')
    parser.add_argument('-lm', '--learning_method', type=str, default="BC",
        help='Choose from BC, GAIL or pass data_generator to save an expert model.')
    parser.add_argument('-pa', '--policy-agent', type=str, default="PPO",
        help='Choose the policy learner agent.')
    parser.add_argument('-df', '--data_file_path', type=str, required=True,
        help='Path to JSON file containing trajectories of a given shape.')
    parser.add_argument('-ms', '--motion-shape', type=str, default="Sine",
        help='Shape of the trajectories as in LASA dataset.')
    parser.add_argument('-nd', '--num-demonstrations', type=int, default=50,
        help='Number of additional demonstrations to the original dataset.')
    parser.add_argument('-neb', '--num-epochs-bc', type=int, default=10,
        help='Number of training epochs for BC.')
    parser.add_argument('-ntg', '--num-timesteps-gail', type=int, default=int(3e4),
        help='Number of training timesteps for GAIL.')
    parser.add_argument('-sp', '--show-plots', action='store_true', default=False,
        help='Show extra plots of final result and trajectories.')
    parser.add_argument('-sm', '--save-model', action='store_true', default=False,
        help='Save the model in the res folder.')
    parser.add_argument('-sd', '--save-dir', type=str,
        default=os.path.join(os.pardir, 'res', 'rlds_policy'),
        help='Optional destination for save/load.')
    parser.add_argument('-mn', '--model-name', type=str, default="test",
        help='Pick a name for the model.')
    parser.add_argument('-wp', '--warm-policy', action='store_true', default=True,
        help='Use a BC policy to kick-start GAIL.')
    args = parser.parse_args()

    if args.learning_method == 'data_generator':
        expert_seds_model(motion_shape=args.motion_shape, save_dir=os.path.join(os.pardir, 'res'))
    else:
        train_rl_policy(learning_method=args.learning_method, policy_agent=args.policy_agent, 
	    motion_shape=args.motion_shape, n_dems=args.num_demonstrations, 
	    n_epochs_bc=args.num_epochs_bc, n_timesteps_gail=args.num_timesteps_gail,
            warm_policy=args.warm_policy, plot=args.show_plots, model_name=args.model_name, 
            save=args.save_model, save_dir=args.save_dir, data_file_path=args.data_file_path)
