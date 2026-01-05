import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import csv
from collections import deque

# --- Network Definition ---


class DQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_q = nn.Linear(128, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_q(x)

# --- Agent Class ---


class DQNAgent:
    def __init__(self, state_dim, action_size, device="cuda"):
        self.device = device
        self.action_size = action_size
        self.batch_size = 64
        self.gamma = 0.99
        self.lr = 0.0005

        self.online_net = DQN(state_dim, action_size).to(device)
        self.target_net = DQN(state_dim, action_size).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_size)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.online_net(state_t).argmax(1).item()

# --- Main Training Loop ---


def run_training():
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = DQNAgent(state_dim, action_dim, device)
    buffer = deque(maxlen=50000)

    log_file = "lunar_lander_dqn_results.csv"
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward', 'Epsilon'])

    # Visualization Setup (Plotting Q-values instead of Distributions)
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    action_labels = ['None', 'Left', 'Main', 'Right']

    epsilon = 1.0
    total_episodes = 1000

    print(f"Starting DQN training on {device}...")

    for episode in range(total_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, epsilon)

            # Visualization: Show Q-values bar chart
            if episode % 10 == 0:
                with torch.no_grad():
                    st = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = agent.online_net(st).squeeze(0).cpu().numpy()
                    ax.clear()
                    ax.bar(action_labels, q_values, color='skyblue')
                    ax.set_title(f"Episode {episode} - Q-Values (ε: {epsilon:.2f})")
                    ax.set_ylabel("Predicted Reward")
                    plt.pause(0.001)

            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            buffer.append((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward

            # Training update
            if len(buffer) > 1000:
                batch = random.sample(buffer, agent.batch_size)
                s, a, r, ns, d = zip(*batch)

                s = torch.FloatTensor(np.array(s)).to(device)
                a = torch.LongTensor(a).to(device).unsqueeze(1)
                r = torch.FloatTensor(r).to(device).unsqueeze(1)
                ns = torch.FloatTensor(np.array(ns)).to(device)
                d = torch.FloatTensor(d).to(device).unsqueeze(1)

                # --- Standard DQN Loss Calculation ---
                with torch.no_grad():
                    # Target: R + gamma * max(Q_target(s'))
                    max_next_q = agent.target_net(ns).max(1)[0].unsqueeze(1)
                    target_q = r + (agent.gamma * max_next_q * (1 - d))

                # Current: Q_online(s, a)
                current_q = agent.online_net(s).gather(1, a)

                loss = F.mse_loss(current_q, target_q)

                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()

        # Decay epsilon
        epsilon = max(0.01, epsilon * 0.995)

        # Update Target Network
        if episode % 5 == 0:
            agent.target_net.load_state_dict(agent.online_net.state_dict())

        # Logging
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, episode_reward, epsilon])

        if episode % 10 == 0:
            print(f"Episode {episode} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.2f}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_training()
