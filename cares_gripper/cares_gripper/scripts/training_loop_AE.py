
import logging
logging.basicConfig(level=logging.INFO)

import os
import cv2
import pydantic
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

import torch
import random
import numpy as np

from environments.RotationEnvironment import RotationEnvironment
from environments.TranslationEnvironment import TranslationEnvironment
from configurations import LearningConfig, EnvironmentConfig, GripperConfig

from AETD3 import AETD3
from cares_reinforcement_learning.memory import MemoryBuffer


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    logging.info("Working with GPU")
else:
    DEVICE = torch.device('cpu')
    logging.info("Working with CPU")


def save_evaluation_values(data_eval_reward, filename):
    data = pd.DataFrame.from_dict(data_eval_reward)
    data.to_csv(f"data_plots/{filename}_evaluation", index=False)
    data.plot(x='step', y='avg_episode_reward', title="Evaluation Reward Curve")
    plt.savefig(f"results_plots/{filename}_evaluation.png")
    plt.close()

def plot_reward_curve(data_reward, filename):
    data = pd.DataFrame.from_dict(data_reward)
    data.to_csv(f"data_plots/{filename}", index=False)
    data.plot(x='step', y='episode_reward', title=filename)
    plt.savefig(f"results_plots/{filename}")
    plt.close()

def plot_reconstruction_img(original, reconstruction):
    input_img      = original[0]/255
    reconstruction = reconstruction[0]
    difference     = abs(input_img - reconstruction)

    plt.subplot(1, 3, 1)
    plt.title("Image Input")
    plt.imshow(input_img, vmin=0, vmax=1)
    plt.subplot(1, 3, 2)
    plt.title("Image Reconstruction")
    plt.imshow(reconstruction, vmin=0, vmax=1)
    plt.subplot(1, 3, 3)
    plt.title("Difference")
    plt.imshow(difference, vmin=0, vmax=1)

    plt.pause(0.01)


def train(environment, agent, memory, learning_config, file_name, intrinsic_on=False):

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    historical_reward = {"step": [], "episode_reward": []}
    historical_reward_evaluation = {"step": [], "avg_episode_reward": []}

    # min_noise    = 0.01
    # noise_decay  = 0.9999
    # noise_scale  = 0.10

    state, goal  = environment.reset()

    for total_step_counter in range(1, int(learning_config.max_steps_training)+1):
        episode_timesteps += 1

        if total_step_counter <= learning_config.max_steps_exploration:
            logging.info(f"Running Exploration Steps {total_step_counter}/{learning_config.max_steps_exploration}")
            action_env = environment.sample_action() # gripper range
            action     = environment.normalize(action_env) # algorithm range [-1, 1]
        else:
            # noise_scale *= noise_decay
            # noise_scale = max(min_noise, noise_scale)
            logging.info(f"Taking step {episode_timesteps} of Episode {episode_num} with Total T {total_step_counter} \n")
            action     = agent.select_action_from_policy(state, goal, noise_scale=0.1)  # algorithm range [-1, 1]
            action_env = environment.denormalize(action)  # gripper range

        (next_state, goal), reward_extrinsic, done, truncated = environment.step(action_env)

        if intrinsic_on and total_step_counter > learning_config.max_steps_exploration:
            alpha = 0.5
            beta  = 0.5
            surprise_rate, novelty_rate = agent.get_intrinsic_value(state, action, next_state)
            reward_surprise = surprise_rate * alpha
            reward_novelty  = novelty_rate * beta
            logging.info(f"Surprise Rate = {reward_surprise},  Novelty Rate = {reward_novelty}, Normal Reward = {reward_extrinsic}")
            total_reward = reward_extrinsic + reward_surprise + reward_novelty

        else:
            total_reward = reward_extrinsic

        memory.add(state=state, action=action, reward=total_reward, next_state=next_state, done=done, goal=goal)
        state = next_state

        episode_reward += reward_extrinsic  # just for plotting and comparing purposes use this reward as it is

        if total_step_counter > learning_config.max_steps_exploration:
            #num_updates = learning_config.max_steps_exploration if total_step_counter == learning_config.max_steps_exploration else learning_config.G
            for _ in range(learning_config.G):
                experience = memory.sample(learning_config.batch_size)
                agent.train_policy((
                    experience['state'],
                    experience['action'],
                    experience['reward'],
                    experience['next_state'],
                    experience['done'],
                    experience['goal']
                ))

            if intrinsic_on:
                agent.train_predictive_model((
                    experience['state'],
                    experience['action'],
                    experience['next_state']
                ))


        if done is True or episode_timesteps >= learning_config.episode_horizont:
            logging.info(f"Total T:{total_step_counter} | Episode {episode_num + 1} was completed with {episode_timesteps} steps taken and a Reward= {episode_reward:.3f}")

            historical_reward["step"].append(total_step_counter)
            historical_reward["episode_reward"].append(episode_reward)

            # Reset environment
            state, goal = environment.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num += 1

            if episode_num % learning_config.plot_freq == 0:
                logging.info("---------- Evaluating Policy and Plotting--------------------")
                plot_reward_curve(historical_reward, file_name)
                evaluation_loop(environment, agent, file_name, learning_config, total_step_counter, historical_reward_evaluation)
                logging.info("-------------------------------------------------------------")

    plot_reward_curve(historical_reward, file_name)
    agent.save_models(file_name)
    logging.info("All GOOD AND DONE :)")



def evaluation_loop(environment, agent, file_name, learning_config, total_counter, historical_reward_evaluation):
    max_steps_evaluation = learning_config.episode_horizont * 5

    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    state, goal = environment.reset()
    frame       = environment.camera.get_frame()

    historical_episode_reward_evaluation = []

    fps = 15
    video_name = f'videos_evaluation/{file_name}_{total_counter}.mp4'
    height, width, channels = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for total_step_counter in range(int(max_steps_evaluation)):
        episode_timesteps += 1
        action     = agent.select_action_from_policy(state, goal,  evaluation=True)  # algorithm range [-1, 1]
        action_env = environment.denormalize(action)  # gripper range
        (state, goal), reward, done, truncated = environment.step(action_env)
        episode_reward += reward

        if episode_num == 0:
            video.write(environment.camera.get_frame())

        if done is True or episode_timesteps >= learning_config.episode_horizont:
            original_img, reconstruction = agent.get_reconstruction_for_evaluation(state)
            plot_reconstruction_img(original_img, reconstruction)

            logging.info(f" EVALUATION | Eval Episode {episode_num + 1} was completed with {episode_timesteps} steps | Reward= {episode_reward:.3f}")
            historical_episode_reward_evaluation.append(episode_reward)

            state, goal = environment.reset()
            episode_reward    = 0
            episode_timesteps = 0
            episode_num       += 1

    mean_reward_evaluation = np.round(np.mean(historical_episode_reward_evaluation), 2)
    historical_reward_evaluation["avg_episode_reward"].append(mean_reward_evaluation)
    historical_reward_evaluation["step"].append(total_counter)
    save_evaluation_values(historical_reward_evaluation, file_name)
    video.release()


def create_directories():
    if not os.path.exists("./results_plots"):
        os.makedirs("./results_plots")
    if not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists("./data_plots"):
        os.makedirs("./data_plots")
    if not os.path.exists("./videos_evaluation"):
        os.makedirs("./videos_evaluation")





def parse_args():
    parser    = ArgumentParser()
    file_path = Path(__file__).parent.resolve()
    parser.add_argument("--learning_config", type=str, default=f"{file_path}/config/learning_config_ID2.json")  # id 2 for robot left
    parser.add_argument("--env_config",      type=str, default=f"{file_path}/config/env_4DOF_config_ID2_AE.json")
    parser.add_argument("--gripper_config",  type=str, default=f"{file_path}/config/gripper_4DOF_config_ID2.json")
    return parser.parse_args()



def main():
    args = parse_args()
    env_config      = pydantic.parse_file_as(path=args.env_config,      type_=EnvironmentConfig)
    gripper_config  = pydantic.parse_file_as(path=args.gripper_config,  type_=GripperConfig)
    learning_config = pydantic.parse_file_as(path=args.learning_config, type_=LearningConfig)

    if env_config.env_type == 0:
        environment = RotationEnvironment(env_config, gripper_config)
    elif env_config.env_type == 1:
        environment = TranslationEnvironment(env_config, gripper_config)

    logging.info("Resetting Environment")
    state, _ = environment.reset()
    logging.info(f" Working with State of Images Shape: {state.shape}")

    action_num = gripper_config.num_motors

    logging.info("Setting up Seeds")
    torch.manual_seed(learning_config.seed)
    np.random.seed(learning_config.seed)
    random.seed(learning_config.seed)

    logging.info("Setting up Memory")
    memory = MemoryBuffer(learning_config.buffer_capacity)

    logging.info("Setting RL Algorithm")

    latent_size = 50
    agent = AETD3(
        latent_size=latent_size,
        action_num=action_num,
        device=DEVICE
    )

    intrinsic_on  = True
    date_time_str = datetime.now().strftime("%m_%d_%H_%M")
    file_name     = f"{date_time_str}_RobotId{gripper_config.gripper_id}_EnvType{env_config.env_type}_ObsType{env_config.observation_type}_Seed{learning_config.seed}_AE_TD3_Intrinsic_{intrinsic_on}"


    create_directories()
    logging.info("Starting Training Loop")
    train(environment, agent, memory, learning_config, file_name, intrinsic_on)


if __name__ == '__main__':
    main()
