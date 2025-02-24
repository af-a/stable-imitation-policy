
"""
Contains general model evaluation functions
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Note: based on plot_trajectories.plot_trajectories()
def generate_trajectories(ds, reference: np.ndarray, space_stretch: float = 0.1,
                          n_samples: int = 1000, file_name: str = "test", save_dir: str = "",
                          n_rollouts: int = 3, rollouts_ic_std: int = 0.1,
                          show_legends: bool = False, save_rollouts: bool = True,
                          test_mean_initial: bool = True):
    """ Execute a policy for given a DS model and save output trajectories.

    Args:
        ds (PlanningPolicyInterface): A dynamical system for motion generation task.
        trajectory (np.ndarray): Input trajectory array (n_samples, dim).
        space_stretch (float, optional): How much of the entire space to show in vector map.
            Defaults to 1.

        file_name(str, optional): Name of the plot file. Defaults to "".
        save_dir(str, optional): Provide a save directory for the figure. Leave empty to
            skip saving. Defaults to "".
        n_samples (int, optional): Number of samples in each demonstration. Defaults to 1000.
        n_rollouts (int, optional): Number of trajectories to reproduce. Defaults to 10.
        show_legends (bool, optional): Opt to show the legends. Defaults to True.
    """

    dim = reference.shape[1]

    # Initial and goal states
    # Note: currently taken directly from training trajectories:
    goal_point = reference[-1]
    initial_states = np.array([reference[idx * n_samples] for idx in range(len(reference) // n_samples)])
    if test_mean_initial:
        initial_states = initial_states.mean(axis=0)[np.newaxis]
    print(f'Initial states ({initial_states.shape}):\n{initial_states}')

    # original policy rollouts
    dt: float = 0.01
    lims = [ (reference[:, i].max() - reference[:, i].min()) for i in range(reference.shape[1])]
    limit = np.linalg.norm(lims) / 100
    # limit = 0.01679588037870918
    limit = 0.017

    simulated_trajs_list = []
    tvec_array_list = []
    traj_generation_time_list = []

    print(f'[INFO] Computed trajectory value limit for generation: {limit}')
    print(f'[INFO] Generating policy rollouts...')
    for idx, start in enumerate(initial_states):
        start_time = time.time()
        print(f'[INFO] Generating trajectory for initial state: {start}')
        simulated_traj: List[np.ndarray] = []
        simulated_traj.append(np.array([start]).reshape(1, dim))

        distance_to_target = np.linalg.norm(simulated_traj[-1] - goal_point)
        while len(simulated_traj) < 5e3:
            if distance_to_target <= limit:
                print(f'[INFO] Reached goal after {len(simulated_traj)} time steps')
                break
            vel = ds.predict(simulated_traj[-1])
            simulated_traj.append(simulated_traj[-1] + dt * vel)
            distance_to_target = np.linalg.norm(simulated_traj[-1] - goal_point)
        else:
            print(f'[INFO] Failed to reach goal after 5000 timesteps. Terminating...')

        tvec_array = np.arange(0., len(simulated_traj) * dt, dt)
        tvec_array_list.append(tvec_array.tolist())

        traj_generation_time_list.append(time.time() - start_time)

        simulated_traj = np.array(simulated_traj)
        simulated_traj = simulated_traj.reshape(simulated_traj.shape[0],
                                                simulated_traj.shape[2])
        simulated_trajs_list.append(simulated_traj)

        if save_rollouts:
            name = file_name if file_name != "" else 'plot'
            os.makedirs(os.path.join(save_dir, name), exist_ok=True)
            np.save(os.path.join(save_dir, name, f'rollout_original_{idx}'), simulated_traj)

    mean_traj_generation_time = sum(traj_generation_time_list) / len(traj_generation_time_list)
    median_traj_generation_time = np.median(traj_generation_time_list)
    print(f'[INFO] Average time to generate a trajectory: {mean_traj_generation_time:.4f} seconds')
    print(f'[INFO] Median time to generate a trajectory: {median_traj_generation_time:.4f} seconds')

    data_dict = {'pos': [traj_array.T.tolist() for traj_array in simulated_trajs_list],
                 'tvec': tvec_array_list,
                 'initial_pos': initial_states.tolist()}

    output_file_path = os.path.join(save_dir, file_name + '.json')
    print(f'[INFO] Saving generated trajectories in: {output_file_path}')
    with open(output_file_path, 'w') as file_handle:
        json.dump(data_dict, file_handle)

def compute_reference_mse(ds, reference_states: np.ndarray, reference_states_der: np.ndarray):
    """ Compute the MSE between reference and predicted velocities at all reference positions.

    Args:
        ds (PlanningPolicyInterface): A dynamical system for motion generation task.
        reference_states (np.ndarray): Reference positions array (n_samples, dim).
        reference_states_der (np.ndarray): Corresponding reference velocities array (n_samples, dim).
    """
    pred_vels = ds.predict(reference_states)

    return ((pred_vels - reference_states_der) ** 2).mean()

