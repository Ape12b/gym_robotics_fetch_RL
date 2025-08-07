import gymnasium as gym
import gymnasium_robotics

def _flatten_obs(obs):
    # If the env returns a dict (e.g., Fetch), concatenate observation + desired_goal
    if isinstance(obs, dict) and 'observation' in obs and 'desired_goal' in obs:
        import numpy as np
        return np.concatenate([obs['observation'], obs['desired_goal']])
    return obs

import torch
import numpy as np
import random
import wandb
import os
import datetime
from td3_torch_p import Agent
from utils import plot_learning_curve


gym.register_envs(gymnasium_robotics)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    config = {
        "env_name": "FetchPickAndPlace-v4",
        "n_epochs": 200, "n_cycles": 50, "n_episodes_per_cycle": 2, "n_train_steps": 40,
        "lr_actor": 1e-3, "lr_critic": 1e-3, "gamma": 0.98, "tau": 0.05,
        "batch_size": 1024, "buffer_capacity": 1e6, "her_ratio": 0.8,
        "policy_delay": 2, "expl_noise": 0.3, "policy_noise": 0.2, "noise_clip": 0.5,
    }
    
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_dir = f"models/{run_id}"; os.makedirs(model_save_dir, exist_ok=True)
    wandb.init(project="TD3-Fetch", config=config, name=f"Run-{run_id}")

    env = gym.make(config['env_name'])
    obs_dim = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
    act_dim = env.action_space.shape[0]; max_action = env.action_space.high[0]
    agent = Agent(alpha=0.001, beta=0.001,
            input_dims=(obs_dim,) , tau=0.005,
            env=env, batch_size=100, layer1_size=400, layer2_size=300,
            n_actions=act_dim)
    
    filename = 'plots/' + 'Robot_fetch_' + config['n_epochs'] + '_games.png'

    best_score = -1000
    score_history = []

    
    total_steps = 0
    for i in range(config['n_epochs']):
        observation = env.reset()
        observation = observation[0]
        done = False
        score = 0
        while not done:
            total_steps += 1
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'average score %.1f' % avg_score)
        wandb.log({"epoch": i, "score": score}, step=total_steps)
    x = [i+1 for i in range(config['n_epochs'])]
    plot_learning_curve(x, score_history, filename)
    env.close(); wandb.finish()
    