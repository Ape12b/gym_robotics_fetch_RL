# import gymnasium as gym
# import gymnasium_robotics
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import random
# from collections import deque
# import wandb # Import wandb
# import os
# import datetime

# # Register the gymnasium-robotics environments
# gym.register_envs(gymnasium_robotics)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # --- Network Architectures ---

# class Actor(nn.Module):
#     def __init__(self, obs_dim, act_dim, max_action):
#         super(Actor, self).__init__()
#         self.max_action = max_action
#         self.fc1 = nn.Linear(obs_dim, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, act_dim)

#     def forward(self, obs):
#         x = F.relu(self.fc1(obs))
#         x = F.relu(self.fc2(x))
#         # Scale the output to [-max_action, max_action]
#         action = self.max_action * torch.tanh(self.fc3(x))
#         return action

# class Critic(nn.Module):
#     def __init__(self, obs_dim, act_dim):
#         super(Critic, self).__init__()
#         # Critic Q1
#         self.fc1 = nn.Linear(obs_dim + act_dim, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 1)

#         # Critic Q2
#         self.fc4 = nn.Linear(obs_dim + act_dim, 256)
#         self.fc5 = nn.Linear(256, 256)
#         self.fc6 = nn.Linear(256, 1)

#     def forward(self, obs, action):
#         obs_action = torch.cat([obs, action], dim=1)
        
#         q1 = F.relu(self.fc1(obs_action))
#         q1 = F.relu(self.fc2(q1))
#         q1 = self.fc3(q1)

#         q2 = F.relu(self.fc4(obs_action))
#         q2 = F.relu(self.fc5(q2))
#         q2 = self.fc6(q2)
        
#         return q1, q2

# # --- Hindsight Experience Replay (HER) Buffer ---

# # --- Hindsight Experience Replay (HER) Buffer ---

# def compute_shaped_reward(obs, next_obs, reward, config):
#     """
#     Computes a robust, hierarchical shaped reward.
#     - Always provides a "reaching" reward for moving the gripper to the object.
#     - Adds a "placing" bonus if the gripper is close, for moving the object to the goal.
#     """
#     if not config["reward_shaping"]:
#         return reward

#     # --- Calculate Reaching Reward component (always active) ---
#     prev_gripper_pos = obs['achieved_goal']
#     prev_obj_pos = obs['observation'][3:6]
#     prev_dist_xy = np.linalg.norm(prev_gripper_pos[:2] - prev_obj_pos[:2])
#     prev_dist_z = np.abs(prev_gripper_pos[2] - prev_obj_pos[2])

#     next_gripper_pos = next_obs['achieved_goal']
#     next_obj_pos = next_obs['observation'][3:6]
#     next_dist_xy = np.linalg.norm(next_gripper_pos[:2] - next_obj_pos[:2])
#     next_dist_z = np.abs(next_gripper_pos[2] - next_obj_pos[2])

#     xy_reward = (prev_dist_xy - next_dist_xy) * config["xy_shaping_scale"]
#     z_reward = (prev_dist_z - next_dist_z) * config["z_shaping_scale"]
#     reaching_reward = xy_reward + z_reward

#     # --- Calculate Placing Reward component (bonus) ---
#     placing_reward = 0  # Initialize bonus to zero
#     # If gripper is close to the block, add a bonus for moving the block to the goal.
#     if next_dist_xy < config["grasp_threshold"]:
#         prev_obj_to_goal = np.linalg.norm(prev_obj_pos - obs['desired_goal'])
#         next_obj_to_goal = np.linalg.norm(next_obj_pos - obs['desired_goal'])
#         # The bonus is the reduction in object-to-goal distance
#         placing_reward = (prev_obj_to_goal - next_obj_to_goal) * (config["z_shaping_scale"] * 2) # Strong bonus

#     # The total shaping reward is the sum of the base reaching reward and the placing bonus
#     shaping_reward = reaching_reward + placing_reward
    
#     return reward + shaping_reward

# class ReplayBuffer:
#     def __init__(self, capacity, her_ratio, env):
#         self.buffer = deque(maxlen=int(capacity))
#         self.her_ratio = her_ratio
#         self.env = env # We need the env to re-compute rewards

#     def add(self, obs, action, reward, next_obs, done):
#         self.buffer.append((obs, action, reward, next_obs, done))

#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         obs_list, action_list, reward_list, next_obs_list, done_list = zip(*batch)

#         # Extract dictionary components into NumPy arrays
#         s = np.array([o['observation'] for o in obs_list])
#         achieved_g = np.array([o['achieved_goal'] for o in obs_list])
#         desired_g = np.array([o['desired_goal'] for o in obs_list])
#         next_s = np.array([o['observation'] for o in next_obs_list])
#         next_achieved_g = np.array([o['achieved_goal'] for o in next_obs_list])
#         action_batch = np.array(action_list)
#         done_batch = np.array(done_list)

#         # --- THIS IS THE CORRECTED REWARD LOGIC ---
#         # Start with the original rewards (which include shaping)
#         reward_batch = np.array(reward_list)

#         # --- HER Goal Relabeling ---
#         her_indices = np.where(np.random.uniform(size=batch_size) < self.her_ratio)
#         future_g = next_achieved_g[her_indices]
#         # Relabel the desired goal for the HER transitions
#         desired_g_her = desired_g.copy() # Use a copy to avoid affecting non-HER transitions
#         desired_g_her[her_indices] = future_g

#         # Re-compute the sparse reward for the HER transitions
#         hindsight_rewards = self.env.compute_reward(next_achieved_g[her_indices], desired_g[her_indices], {})
        
#         # Replace the original shaped reward with the new hindsight reward ONLY for HER transitions
#         reward_batch[her_indices] = hindsight_rewards
#         # --- End of HER Correction ---

#         # Concatenate state and goal for network input
#         s_input = np.concatenate([s, desired_g_her], axis=1) # Use the relabeled goals
#         s_next_input = np.concatenate([next_s, desired_g_her], axis=1)

#         # Convert to tensors
#         s_tensor = torch.tensor(s_input, dtype=torch.float32, device=device)
#         a_tensor = torch.tensor(action_batch, dtype=torch.float32, device=device)
#         r_tensor = torch.tensor(reward_batch, dtype=torch.float32, device=device).unsqueeze(1)
#         d_tensor = torch.tensor(done_batch, dtype=torch.float32, device=device).unsqueeze(1)
#         s_next_tensor = torch.tensor(s_next_input, dtype=torch.float32, device=device)

#         return s_tensor, a_tensor, r_tensor, d_tensor, s_next_tensor

#     def __len__(self):
#         return len(self.buffer)  
# # --- TD3 Agent ---

# class TD3Agent:
#     def __init__(self, obs_dim, act_dim, max_action, config):
#         self.config = config
#         self.act_dim = act_dim
#         self.max_action = max_action

#         self.actor = Actor(obs_dim, act_dim, max_action).to(device)
#         self.actor_target = Actor(obs_dim, act_dim, max_action).to(device)
#         self.actor_target.load_state_dict(self.actor.state_dict())
#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['lr_actor'])

#         self.critic = Critic(obs_dim, act_dim).to(device)
#         self.critic_target = Critic(obs_dim, act_dim).to(device)
#         self.critic_target.load_state_dict(self.critic.state_dict())
#         self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr_critic'])
        
#         self.total_it = 0

#     def select_action(self, obs_dict, add_noise=True):
#         obs = np.concatenate([obs_dict['observation'], obs_dict['desired_goal']])
#         obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        
#         with torch.no_grad():
#             action = self.actor(obs_tensor).squeeze(0).cpu().numpy()

#         if add_noise:
#             noise = np.random.normal(0, self.max_action * self.config['expl_noise'], size=self.act_dim)
#             action = action + noise
            
#         return np.clip(action, -self.max_action, self.max_action)

#     def update(self, replay_buffer):
#         self.total_it += 1
#         s, a, r, d, s_next = replay_buffer.sample(self.config['batch_size'])

#         with torch.no_grad():
#             noise = (torch.randn_like(a) * self.config['policy_noise']).clamp(
#                 -self.config['noise_clip'], self.config['noise_clip']
#             )
#             a_next = (self.actor_target(s_next) + noise).clamp(-self.max_action, self.max_action)
#             q1_next, q2_next = self.critic_target(s_next, a_next)
#             q_next = torch.min(q1_next, q2_next)
#             y = r + self.config['gamma'] * (1.0 - d) * q_next

#         q1, q2 = self.critic(s, a)
#         critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        
#         self.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         self.critic_optimizer.step()
        
#         actor_loss_val = None
#         if self.total_it % self.config['policy_delay'] == 0:
#             a_pred = self.actor(s)
#             q_pred, _ = self.critic(s, a_pred)
#             actor_loss = -q_pred.mean()
#             actor_loss_val = actor_loss.item()
            
#             self.actor_optimizer.zero_grad()
#             actor_loss.backward()
#             self.actor_optimizer.step()
            
#             self._soft_update(self.actor_target, self.actor)
#             self._soft_update(self.critic_target, self.critic)
        
#         return critic_loss.item(), actor_loss_val
            
#     def _soft_update(self, target, source):
#         tau = self.config['tau']
#         for t_param, s_param in zip(target.parameters(), source.parameters()):
#             t_param.data.copy_(tau * s_param.data + (1.0 - tau) * t_param.data)

# # --- Main Training Loop ---
# if __name__ == "__main__":
    
#     config = {
#         "env_name": "FetchPickAndPlace-v4",
#         "num_episodes": 30000, # You might need more episodes for good performance
#         "num_time_steps": 100,
#         "lr_actor": 1e-4,
#         "lr_critic": 1e-3,
#         "gamma": 0.98,
#         "tau": 0.005,
#         "batch_size": 1024,
#         "buffer_capacity": 1e6,
#         "her_ratio": 0.8,
#         "policy_delay": 2,
#         "expl_noise": 0.25,
#         "policy_noise": 0.2,
#         "noise_clip": 0.5,
#         "reward_shaping": True,
#         # "shaping_scale": 100, # Using the stronger scale,
#         "xy_shaping_scale": 10.0,  # Scale for horizontal movement
#         "z_shaping_scale": 50.0,   # STRONGER scale for vertical movement
#         "grasp_threshold": 0.04
#     }

#     # --- 1. CREATE UNIQUE DIRECTORY FOR SAVING MODELS ---
#     run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     model_save_dir = f"models/{run_id}"
#     os.makedirs(model_save_dir, exist_ok=True)

#     wandb.init(
#         project="TD3-HER-Fetch", 
#         config=config,
#         name=f"TD3-HER-Shaped-{run_id}" # Use run_id for wandb name
#     )

#     env = gym.make(config['env_name'])
#     obs_dict = env.reset()[0]
#     obs_dim = obs_dict['observation'].shape[0] + obs_dict['desired_goal'].shape[0]
#     act_dim = env.action_space.shape[0]
#     max_action = env.action_space.high[0]

#     agent = TD3Agent(obs_dim, act_dim, max_action, config)
#     replay_buffer = ReplayBuffer(config['buffer_capacity'], config['her_ratio'], env.unwrapped)

#     print(f"Starting training on {device}...")
#     print(f"Models will be saved in: {model_save_dir}")
#     print(f"Track progress at: {wandb.run.url}")

#     success_queue = deque(maxlen=100)
#     best_success_rate = -1.0 # --- 2. INITIALIZE BEST SUCCESS RATE TRACKER ---

#     for ep in range(config['num_episodes']):
#         obs, _ = env.reset()
#         ep_reward_sparse, ep_shaped_reward = 0, 0
#         ep_success = False

#         for t in range(config['num_time_steps']):
#             action = agent.select_action(obs)
#             next_obs, reward, terminated, truncated, info = env.step(action)
#             done = terminated or truncated
#             shaped_reward = compute_shaped_reward(obs, next_obs, reward, config)
#             replay_buffer.add(obs, action, shaped_reward, next_obs, done)

#             if len(replay_buffer) > config['batch_size']:
#                 critic_loss, actor_loss = agent.update(replay_buffer)
#                 log_data = {"critic_loss": critic_loss}
#                 if actor_loss is not None:
#                     log_data["actor_loss"] = actor_loss
#                 wandb.log(log_data)

#             obs = next_obs
#             ep_reward_sparse += reward
#             ep_shaped_reward += shaped_reward
#             if info.get('is_success', False):
#                 ep_success = True
#             if done:
#                 break
        
#         success_queue.append(1.0 if ep_success else 0.0)
#         current_avg_success_rate = np.mean(success_queue)

#         wandb.log({
#             "episode": ep + 1,
#             "episode_reward_sparse": ep_reward_sparse,
#             "episode_reward_shaped": ep_shaped_reward,
#             "success_rate_100_eps": current_avg_success_rate
#         })
        
#         # --- 3. CHECK AND SAVE THE BEST MODEL ---
#         if len(success_queue) == 100 and current_avg_success_rate > best_success_rate:
#             best_success_rate = current_avg_success_rate
#             print(f"** New best model saved with success rate: {best_success_rate:.2f} at episode {ep+1} **")
#             torch.save(agent.actor.state_dict(), f"{model_save_dir}/actor_best.pth")

#         if (ep + 1) % 50 == 0:
#             print(f"Episode {ep+1}/{config['num_episodes']} | Avg Success Rate (last 100): {current_avg_success_rate:.2f}")

#     env.close()
#     wandb.finish()
#     print("Training finished.")

# import gymnasium as gym
# import gymnasium_robotics
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import random
# from collections import deque
# import wandb
# import os
# import datetime

# gym.register_envs(gymnasium_robotics)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # --- 1. NEW: Observation Normalizer ---
# class Normalizer:
#     def __init__(self, size, eps=1e-2):
#         self.size = size
#         self.eps = eps
#         self.sum = np.zeros(self.size, np.float32)
#         self.sumsq = np.zeros(self.size, np.float32)
#         self.count = 0
#         self.mean = np.zeros(self.size, np.float32)
#         self.std = np.ones(self.size, np.float32)

#     def update(self, v):
#         self.sum += v.sum(axis=0)
#         self.sumsq += (np.square(v)).sum(axis=0)
#         self.count += v.shape[0]
#         self.mean = self.sum / self.count
#         self.std = np.sqrt(np.maximum(np.square(self.eps), (self.sumsq / self.count) - np.square(self.mean)))

#     def normalize(self, v):
#         return (v - self.mean) / self.std

# # --- Network Architectures ---
# class Actor(nn.Module):
#     def __init__(self, obs_dim, act_dim, max_action):
#         super(Actor, self).__init__()
#         self.max_action = max_action
#         self.fc1 = nn.Linear(obs_dim, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, act_dim)

#     def forward(self, obs):
#         x = F.relu(self.fc1(obs))
#         x = F.relu(self.fc2(x))
#         action = self.max_action * torch.tanh(self.fc3(x))
#         return action

# class Critic(nn.Module):
#     def __init__(self, obs_dim, act_dim):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(obs_dim + act_dim, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 1)

#         self.fc4 = nn.Linear(obs_dim + act_dim, 256)
#         self.fc5 = nn.Linear(256, 256)
#         self.fc6 = nn.Linear(256, 1)

#     def forward(self, obs, action):
#         obs_action = torch.cat([obs, action], dim=1)
#         q1 = F.relu(self.fc1(obs_action)); q1 = F.relu(self.fc2(q1)); q1 = self.fc3(q1)
#         q2 = F.relu(self.fc4(obs_action)); q2 = F.relu(self.fc5(q2)); q2 = self.fc6(q2)
#         return q1, q2

# # --- Replay Buffer with HER (Now using sparse rewards) ---
# class ReplayBuffer:
#     def __init__(self, capacity, her_ratio, env):
#         self.buffer = deque(maxlen=int(capacity))
#         self.her_ratio = her_ratio
#         self.env = env

#     def add(self, obs, action, reward, next_obs, done):
#         self.buffer.append((obs, action, reward, next_obs, done))

#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         obs_list, action_list, reward_list, next_obs_list, done_list = zip(*batch)
        
#         s = np.array([o['observation'] for o in obs_list])
#         achieved_g = np.array([o['achieved_goal'] for o in obs_list])
#         desired_g = np.array([o['desired_goal'] for o in obs_list])
#         next_s = np.array([o['observation'] for o in next_obs_list])
#         next_achieved_g = np.array([o['achieved_goal'] for o in next_obs_list])
        
#         her_indices = np.where(np.random.uniform(size=batch_size) < self.her_ratio)
#         future_g = next_achieved_g[her_indices]
#         desired_g[her_indices] = future_g
        
#         reward_batch = self.env.compute_reward(next_achieved_g, desired_g, {})
        
#         obs_batch = np.concatenate([s, desired_g], axis=1)
#         next_obs_batch = np.concatenate([next_s, desired_g], axis=1)
#         action_batch = np.array(action_list)
#         done_batch = np.array(done_list)

#         return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

#     def __len__(self):
#         return len(self.buffer)
        
# # --- TD3 Agent with Normalizer ---
# class TD3Agent:
#     def __init__(self, obs_dim, act_dim, max_action, config):
#         self.config = config
#         self.act_dim = act_dim
#         self.max_action = max_action

#         self.actor = Actor(obs_dim, act_dim, max_action).to(device)
#         self.actor_target = Actor(obs_dim, act_dim, max_action).to(device)
#         self.actor_target.load_state_dict(self.actor.state_dict())
#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['lr_actor'])

#         self.critic = Critic(obs_dim, act_dim).to(device)
#         self.critic_target = Critic(obs_dim, act_dim).to(device)
#         self.critic_target.load_state_dict(self.critic.state_dict())
#         self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr_critic'])
        
#         # --- 2. ADD NORMALIZER TO THE AGENT ---
#         self.obs_normalizer = Normalizer(obs_dim)
        
#     def select_action(self, obs_dict, add_noise=True):
#         obs_cat = np.concatenate([obs_dict['observation'], obs_dict['desired_goal']])
        
#         # --- 3. NORMALIZE OBSERVATION BEFORE USING IT ---
#         obs_norm = self.obs_normalizer.normalize(obs_cat)
#         obs_tensor = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0).to(device)
        
#         with torch.no_grad():
#             action = self.actor(obs_tensor).squeeze(0).cpu().numpy()

#         if add_noise:
#             action += self.config['expl_noise'] * np.random.randn(self.act_dim)
            
#         return np.clip(action, -self.max_action, self.max_action)

#     def update(self, replay_buffer):
#         actor_losses, critic_losses = [], []
#         # --- 4. INTENSE TRAINING LOOP (CYCLES) ---
#         for _ in range(self.config['n_train_steps']):
#             s, a, r, s_next, d = replay_buffer.sample(self.config['batch_size'])
            
#             # --- 5. NORMALIZE BATCHED DATA ---
#             s_norm = self.obs_normalizer.normalize(s)
#             s_next_norm = self.obs_normalizer.normalize(s_next)
            
#             s_norm = torch.tensor(s_norm, dtype=torch.float32, device=device)
#             a = torch.tensor(a, dtype=torch.float32, device=device)
#             r = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
#             d = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)
#             s_next_norm = torch.tensor(s_next_norm, dtype=torch.float32, device=device)

#             with torch.no_grad():
#                 noise = (torch.randn_like(a) * self.config['policy_noise']).clamp(-self.config['noise_clip'], self.config['noise_clip'])
#                 a_next = (self.actor_target(s_next_norm) + noise).clamp(-self.max_action, self.max_action)
#                 q1_next, q2_next = self.critic_target(s_next_norm, a_next)
#                 q_next = torch.min(q1_next, q2_next)
#                 y = r + self.config['gamma'] * (1.0 - d) * q_next

#             q1, q2 = self.critic(s_norm, a)
#             critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            
#             self.critic_optimizer.zero_grad(); critic_loss.backward(); self.critic_optimizer.step()
#             critic_losses.append(critic_loss.item())
            
#             # Delayed Policy Updates
#             if len(critic_losses) % self.config['policy_delay'] == 0:
#                 a_pred = self.actor(s_norm)
#                 q_pred, _ = self.critic(s_norm, a_pred)
#                 actor_loss = -q_pred.mean()
                
#                 self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
#                 actor_losses.append(actor_loss.item())

#                 # Soft Update Target Networks
#                 for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
#                     target_param.data.copy_(param.data * self.config['tau'] + target_param.data * (1.0 - self.config['tau']))
#                 for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
#                     target_param.data.copy_(param.data * self.config['tau'] + target_param.data * (1.0 - self.config['tau']))
        
#         return np.mean(critic_losses), np.mean(actor_losses)

# # --- Main Training Loop ---
# if __name__ == "__main__":
#     config = {
#         "env_name": "FetchPickAndPlace-v4",
#         "n_epochs": 100, # Number of major training cycles
#         "n_cycles": 50,  # Cycles per epoch
#         "n_episodes_per_cycle": 2, # Episodes to collect data per cycle
#         "n_train_steps": 40, # Gradient steps per cycle
#         "lr_actor": 1e-3, "lr_critic": 1e-3,
#         "gamma": 0.98, "tau": 0.05,
#         "batch_size": 1024, "buffer_capacity": 1e6,
#         "her_ratio": 0.8, "policy_delay": 2,
#         "expl_noise": 0.2, "policy_noise": 0.2, "noise_clip": 0.5
#     }
    
#     run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     wandb.init(project="TD3-HER-Fetch-Final", config=config, name=f"Run-{run_id}")

#     env = gym.make(config['env_name'])
#     obs_dict = env.reset()[0]
#     obs_dim = obs_dict['observation'].shape[0] + obs_dict['desired_goal'].shape[0]
#     act_dim = env.action_space.shape[0]
#     max_action = env.action_space.high[0]

#     agent = TD3Agent(obs_dim, act_dim, max_action, config)
#     replay_buffer = ReplayBuffer(config['buffer_capacity'], config['her_ratio'], env.unwrapped)

#     print(f"Starting training...")
#     total_steps = 0
#     for epoch in range(config['n_epochs']):
#         epoch_successes = []
#         for cycle in range(config['n_cycles']):
#             # Collect experience
#             for _ in range(config['n_episodes_per_cycle']):
#                 obs, _ = env.reset()
#                 for t in range(env._max_episode_steps):
#                     action = agent.select_action(obs)
#                     next_obs, reward, terminated, truncated, info = env.step(action)
#                     replay_buffer.add(obs, action, reward, next_obs, terminated) # Use terminated, not done
#                     obs = next_obs
#                     total_steps += 1
            
#             # Train agent
#             if len(replay_buffer) > config['batch_size']:
#                 c_loss, a_loss = agent.update(replay_buffer)
#                 wandb.log({"critic_loss": c_loss, "actor_loss": a_loss, "step": total_steps})

#             # Update normalizer stats with all data in buffer
#             # This is a simplification; a better way is to update with recent data
#             all_transitions = list(replay_buffer.buffer)
#             obs_list, _, _, next_obs_list, _ = zip(*all_transitions)
#             s = np.array([o['observation'] for o in obs_list])
#             g = np.array([o['desired_goal'] for o in obs_list])
#             obs_buffer = np.concatenate([s,g], axis=1)
#             agent.obs_normalizer.update(obs_buffer)

#         # Evaluate agent performance at the end of each epoch
#         success_rate = 0
#         for _ in range(10): # 10 evaluation episodes
#             obs, _ = env.reset()
#             ep_success = False
#             for t in range(env._max_episode_steps):
#                 action = agent.select_action(obs, add_noise=False) # Deterministic
#                 obs, _, _, _, info = env.step(action)
#                 if info.get('is_success'):
#                     ep_success = True
#                     break
#             if ep_success: success_rate += 1
        
#         success_rate /= 10
#         print(f"Epoch: {epoch+1}/{config['n_epochs']} | Success Rate: {success_rate:.2f}")
#         wandb.log({"epoch": epoch, "success_rate": success_rate, "step": total_steps})

#     env.close()
#     wandb.finish()
import gymnasium as gym
import gymnasium_robotics
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import wandb
import os
import datetime

gym.register_envs(gymnasium_robotics)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Normalizer:
    def __init__(self, size, eps=1e-2):
        self.size = size
        self.eps = eps
        self.sum = np.zeros(self.size, np.float32)
        self.sumsq = np.zeros(self.size, np.float32)
        self.count = 0
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)

    def update(self, v):
        self.sum += v.sum(axis=0)
        self.sumsq += (np.square(v)).sum(axis=0)
        self.count += v.shape[0]
        self.mean = self.sum / self.count
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.sumsq / self.count) - np.square(self.mean)))

    def normalize(self, v):
        return (v - self.mean) / self.std

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, act_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs)); x = F.relu(self.fc2(x)); action = self.max_action * torch.tanh(self.fc3(x))
        return action

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, 256); self.fc2 = nn.Linear(256, 256); self.fc3 = nn.Linear(256, 1)
        self.fc4 = nn.Linear(obs_dim + act_dim, 256); self.fc5 = nn.Linear(256, 256); self.fc6 = nn.Linear(256, 1)

    def forward(self, obs, action):
        obs_action = torch.cat([obs, action], dim=1)
        q1 = F.relu(self.fc1(obs_action)); q1 = F.relu(self.fc2(q1)); q1 = self.fc3(q1)
        q2 = F.relu(self.fc4(obs_action)); q2 = F.relu(self.fc5(q2)); q2 = self.fc6(q2)
        return q1, q2

class ReplayBuffer:
    def __init__(self, capacity, her_ratio, env):
        self.buffer = deque(maxlen=int(capacity))
        self.her_ratio = her_ratio
        self.env = env

    def add(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs_list, action_list, reward_list, next_obs_list, done_list = zip(*batch)
        
        s = np.array([o['observation'] for o in obs_list])
        achieved_g = np.array([o['achieved_goal'] for o in obs_list])
        desired_g = np.array([o['desired_goal'] for o in obs_list])
        next_s = np.array([o['observation'] for o in next_obs_list])
        next_achieved_g = np.array([o['achieved_goal'] for o in next_obs_list])
        
        her_indices = np.where(np.random.uniform(size=batch_size) < self.her_ratio)
        future_g = next_achieved_g[her_indices]
        desired_g[her_indices] = future_g
        
        reward_batch = self.env.compute_reward(next_achieved_g, desired_g, {})
        
        obs_batch = np.concatenate([s, desired_g], axis=1)
        next_obs_batch = np.concatenate([next_s, desired_g], axis=1)
        action_batch = np.array(action_list)
        done_batch = np.array(done_list)

        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def __len__(self):
        return len(self.buffer)

class TD3Agent:
    def __init__(self, obs_dim, act_dim, max_action, config):
        self.config = config; self.act_dim = act_dim; self.max_action = max_action
        self.actor = Actor(obs_dim, act_dim, max_action).to(device)
        self.actor_target = Actor(obs_dim, act_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['lr_actor'])
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.critic_target = Critic(obs_dim, act_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr_critic'])
        self.obs_normalizer = Normalizer(obs_dim)
        
    def select_action(self, obs_dict, add_noise=True):
        obs_cat = np.concatenate([obs_dict['observation'], obs_dict['desired_goal']])
        obs_norm = self.obs_normalizer.normalize(obs_cat)
        obs_tensor = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(obs_tensor).squeeze(0).cpu().numpy()
        if add_noise:
            action += self.config['expl_noise'] * np.random.randn(self.act_dim)
        return np.clip(action, -self.max_action, self.max_action)

    def update(self, replay_buffer, total_updates):
        actor_losses, critic_losses = [], []
        for i in range(self.config['n_train_steps']):
            s, a, r, s_next, d = replay_buffer.sample(self.config['batch_size'])
            s_norm = self.obs_normalizer.normalize(s)
            s_next_norm = self.obs_normalizer.normalize(s_next)
            s_norm_th = torch.tensor(s_norm, dtype=torch.float32, device=device)
            a_th = torch.tensor(a, dtype=torch.float32, device=device)
            r_th = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
            d_th = torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1)
            s_next_norm_th = torch.tensor(s_next_norm, dtype=torch.float32, device=device)

            with torch.no_grad():
                noise = (torch.randn_like(a_th) * self.config['policy_noise']).clamp(-self.config['noise_clip'], self.config['noise_clip'])
                a_next = (self.actor_target(s_next_norm_th) + noise).clamp(-self.max_action, self.max_action)
                q1_next, q2_next = self.critic_target(s_next_norm_th, a_next)
                q_next = torch.min(q1_next, q2_next)
                y = r_th + self.config['gamma'] * (1.0 - d_th) * q_next

            q1, q2 = self.critic(s_norm_th, a_th)
            critic_loss = F.mse_loss(y, q1) + F.mse_loss(y, q2)
            self.critic_optimizer.zero_grad(); critic_loss.backward(); self.critic_optimizer.step()
            critic_losses.append(critic_loss.item())
            
            if (total_updates + i) % self.config['policy_delay'] == 0:
                a_pred = self.actor(s_norm_th)
                q_pred, _ = self.critic(s_norm_th, a_pred)
                actor_loss = -q_pred.mean()
                self.actor_optimizer.zero_grad(); actor_loss.backward(); self.actor_optimizer.step()
                actor_losses.append(actor_loss.item())

                for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                    target_param.data.copy_(param.data * self.config['tau'] + target_param.data * (1.0 - self.config['tau']))
                for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                    target_param.data.copy_(param.data * self.config['tau'] + target_param.data * (1.0 - self.config['tau']))
        return np.mean(critic_losses), np.mean(actor_losses) if actor_losses else None

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
    wandb.init(project="TD3-HER-Fetch-Robust", config=config, name=f"Run-{run_id}")

    env = gym.make(config['env_name'])
    obs_dim = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
    act_dim = env.action_space.shape[0]; max_action = env.action_space.high[0]

    agent = TD3Agent(obs_dim, act_dim, max_action, config)
    replay_buffer = ReplayBuffer(config['buffer_capacity'], config['her_ratio'], env.unwrapped)

    print(f"Starting training... Models will be saved in: {model_save_dir}")
    total_updates = 0; best_success_rate = -1.0

    for epoch in range(config['n_epochs']):
        cycle_obs = []
        for cycle in range(config['n_cycles']):
            # Collect experience
            for _ in range(config['n_episodes_per_cycle']):
                obs, _ = env.reset()
                for t in range(env._max_episode_steps):
                    action = agent.select_action(obs)
                    next_obs, reward, terminated, _, info = env.step(action)
                    replay_buffer.add(obs, action, reward, next_obs, terminated)
                    cycle_obs.append(np.concatenate([obs['observation'], obs['desired_goal']]))
                    obs = next_obs
            
            # Train agent and get losses
            if len(replay_buffer) > config['batch_size']:
                c_loss, a_loss = agent.update(replay_buffer, total_updates)
                total_updates += config['n_train_steps']
                if a_loss is not None:
                    wandb.log({"critic_loss": c_loss, "actor_loss": a_loss}, step=total_updates)
        
        # Update normalizer with data from the epoch
        agent.obs_normalizer.update(np.array(cycle_obs))

        # Evaluate agent performance at the end of each epoch
        success_rate = 0
        for _ in range(10): # 10 evaluation episodes
            obs, _ = env.reset()
            for t in range(env._max_episode_steps):
                action = agent.select_action(obs, add_noise=False) # Deterministic
                obs, _, _, _, info = env.step(action)
                if info.get('is_success'):
                    success_rate += 1; break
        success_rate /= 10

        if success_rate > best_success_rate:
            best_success_rate = success_rate
            print(f"** Epoch {epoch+1}: New best model saved with success rate: {best_success_rate:.2f} **")
            torch.save(agent.actor.state_dict(), f"{model_save_dir}/actor_best.pth")

        print(f"Epoch: {epoch+1}/{config['n_epochs']} | Success Rate: {success_rate:.2f}")
        wandb.log({"epoch": epoch, "success_rate": success_rate}, step=total_updates)

    env.close(); wandb.finish()