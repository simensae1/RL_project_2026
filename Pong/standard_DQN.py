import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformReward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import argparse
import csv
import ale_py
from collections import deque

# Register ALE environments
gym.register_envs(ale_py)

# --- Network Definition (Standard Scalar DQN) ---


class StandardDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(StandardDQN, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        fc_input_dim = self.feature_size(input_shape)
        self.fc1 = nn.Linear(fc_input_dim, 512)
        self.fc_q = nn.Linear(512, num_actions)  # Outputs 1 scalar per action

    def feature_size(self, input_shape):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *input_shape)))).view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc_q(x)

# --- Agent Class ---


class DQNAgent:
    def __init__(self, state_shape, action_size, device="cuda"):
        self.device = device
        self.action_size = action_size
        self.batch_size = 32
        self.gamma = 0.99
        self.learning_rate = 0.00025
        self.update_target_freq = 10000

        self.online_net = StandardDQN(state_shape, action_size).to(device)
        self.target_net = StandardDQN(state_shape, action_size).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.steps = 0

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t)
            return q_values.argmax(dim=1).item()

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (Standard Bellman Equation)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Loss function (MSE instead of KL)
        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.update_target_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return loss.item()

# --- Replay Buffer ---


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return np.array(s), a, r, np.array(ns), d

# --- Main Training Loop ---


def run_training(game_id, seed, total_frames=11_000_000):
    env = gym.make(game_id, render_mode=None, frameskip=1)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4, scale_obs=False)
    env = TransformReward(env, lambda r: np.sign(r))
    env = FrameStackObservation(env, stack_size=4)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DQNAgent((4, 84, 84), env.action_space.n, device)
    buffer = ReplayBuffer(capacity=1000000)

    log_filename = f"results_{game_id.split('/')[-1]}_standardDQN_seed{seed}.csv"
    with open(log_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'score', 'loss'])

    epsilon = 0.05
    state, _ = env.reset(seed=seed)
    state = np.array(state)
    episode_reward = 0
    recent_loss = 0

    print(f"Starting standard DQN training on {game_id}. Device: {device}")

    for frame_idx in range(1, total_frames + 1):
        action = agent.select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = np.array(next_state)

        buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if len(buffer) > 1000 and frame_idx % 4 == 0:
            states, actions, rewards, next_states, dones = buffer.sample(32)
            recent_loss = agent.train_step(states, actions, rewards, next_states, dones)

        if done:
            with open(log_filename, 'a', newline='') as f:
                csv.writer(f).writerow([frame_idx, episode_reward, recent_loss])
            print(f"Frame: {frame_idx}, Score: {episode_reward}, Loss: {recent_loss:.4f}")
            episode_reward = 0
            state, _ = env.reset()
            state = np.array(state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="ALE/Pong-v5")
    parser.add_argument("--frames", type=int, default=11_000_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_training(args.game, args.seed, args.frames)
