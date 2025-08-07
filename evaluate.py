import gymnasium as gym
import gymnasium_robotics
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
from pathlib import Path

# Register the gymnasium-robotics environments
gym.register_envs(gymnasium_robotics)

# --- IMPORTANT: You must define the EXACT same Actor class as in your training script ---
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, act_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        action = self.max_action * torch.tanh(self.fc3(x))
        return action

def run_evaluation(model_path: str, mode: str, num_episodes: int = 10):
    """
    Loads a saved actor model and runs it in the environment.
    - 'human' mode: Renders to a pop-up window.
    - 'video' mode: Saves an mp4 file.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model path not found at {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Setup Environment based on selected mode ---
    if mode == 'video':
        print("Mode: Saving video. No pop-up window will be shown.")
        # To save a video, we MUST use 'rgb_array' render mode
        env = gym.make("FetchPickAndPlace-v4", render_mode="rgb_array")
        
        # Extract the run_id from the model path to create a matching video folder
        try:
            run_id = Path(model_path).parent.name
            video_folder = f"videos/{run_id}"
        except Exception:
            video_folder = "videos/latest_run"
        print(f"Saving videos to: {video_folder}")
        env = gym.wrappers.RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True)
    
    elif mode == 'human':
        print("Mode: Human visualization. A window will pop up.")
        env = gym.make("FetchPickAndPlace-v4", render_mode="human")
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'human' or 'video'.")

    # Get environment parameters
    obs_dict = env.reset()[0]
    obs_dim = obs_dict['observation'].shape[0] + obs_dict['desired_goal'].shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # --- 2. Initialize Actor and Load Saved Weights ---
    actor = Actor(obs_dim, act_dim, max_action).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval() # Set model to evaluation mode

    print("Starting evaluation...")
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            # --- 3. Select Action Deterministically (no exploration noise) ---
            obs_input = np.concatenate([obs['observation'], obs['desired_goal']])
            obs_tensor = torch.tensor(obs_input, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = actor(obs_tensor).squeeze(0).cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            if info.get('is_success'):
                print(f"Episode {ep+1}: SUCCESS!")
    
    print("Evaluation finished.")
    env.close()

if __name__ == "__main__":
    # Use argparse to specify the model path and mode from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved actor .pth file")
    parser.add_argument("--mode", type=str, default="human", choices=['human', 'video'], help="Evaluation mode: 'human' for live viewing, 'video' for saving an mp4.")
    args = parser.parse_args()
    
    run_evaluation(args.model_path, args.mode)