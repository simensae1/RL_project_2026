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


class CategoricalDQN(nn.Module):
    def __init__(self, input_dim, num_actions, num_atoms=51, Vmin=-150, Vmax=250):
        super(CategoricalDQN, self).__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax

        self.register_buffer("support", torch.linspace(Vmin, Vmax, num_atoms))

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_q = nn.Linear(128, num_actions * num_atoms)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_logits = self.fc_q(x).view(-1, self.num_actions, self.num_atoms)
        return F.log_softmax(q_logits, dim=2)

    def get_q_value(self, x):
        log_probs = self.forward(x)
        probs = log_probs.exp()
        return (probs * self.support).sum(dim=2)

# --- Projection Helper ---


def projection_distribution(next_dist, rewards, dones, gamma, num_atoms, Vmin, Vmax, support):
    batch_size = rewards.size(0)
    delta_z = (Vmax - Vmin) / (num_atoms - 1)

    Tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * gamma * support.unsqueeze(0)
    Tz = Tz.clamp(min=Vmin, max=Vmax)
    b = (Tz - Vmin) / delta_z
    l, u = b.floor().long(), b.ceil().long()

    m = torch.zeros(batch_size, num_atoms).to(rewards.device)
    offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long().unsqueeze(1).expand(batch_size, num_atoms).to(rewards.device)

    m.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    m.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
    return m

# --- Agent Class ---


class C51Agent:
    def __init__(self, state_dim, action_size, device="cuda"):
        self.device = device
        self.action_size = action_size
        self.num_atoms = 51
        self.Vmin, self.Vmax = -150, 250
        self.batch_size = 64
        self.gamma = 0.99
        self.lr = 0.0005

        self.online_net = CategoricalDQN(state_dim, action_size, self.num_atoms, self.Vmin, self.Vmax).to(device)
        self.target_net = CategoricalDQN(state_dim, action_size, self.num_atoms, self.Vmin, self.Vmax).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.online_net.get_q_value(state_t).argmax(1).item()

# --- Main Training Loop ---


def run_training():
    # Initialization
    env = gym.make("LunarLander-v3", render_mode="human")  # Change to "human" to watch the lander
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = C51Agent(state_dim, action_dim, device)
    buffer = deque(maxlen=50000)

    # Logging Setup
    log_file = "lunar_lander_c51_results.csv"
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward', 'Epsilon'])

    # Visualization Setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    action_labels = ['None', 'Left', 'Main', 'Right']

    epsilon = 1.0
    total_episodes = 1000

    print(f"Starting training on {device}...")

    for episode in range(total_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, epsilon)

            # Real-time Visualization (every 20 steps to save CPU)
            if episode % 10 == 0:
                with torch.no_grad():
                    st = torch.FloatTensor(state).unsqueeze(0).to(device)
                    probs = agent.online_net(st).exp().squeeze(0).cpu().numpy()
                    ax.clear()
                    for i in range(action_dim):
                        ax.plot(agent.online_net.support.cpu().numpy(), probs[i], label=action_labels[i])
                    ax.set_title(f"Episode {episode} - Prob. Distribution (ε: {epsilon:.2f})")
                    ax.set_ylim(0, 0.5)  # Fix y-axis to see the spikes better
                    ax.legend()
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
                a = torch.LongTensor(a).to(device)
                r = torch.FloatTensor(r).to(device)
                ns = torch.FloatTensor(np.array(ns)).to(device)
                d = torch.FloatTensor(d).to(device)

                with torch.no_grad():
                    next_actions = agent.target_net.get_q_value(ns).argmax(1)
                    next_probs = agent.target_net(ns).exp()
                    next_best_dist = next_probs[range(agent.batch_size), next_actions]
                    target_dist = projection_distribution(next_best_dist, r, d, agent.gamma, agent.num_atoms, agent.Vmin, agent.Vmax, agent.online_net.support)

                log_probs = agent.online_net(s)
                pred_dist = log_probs[range(agent.batch_size), a]
                loss = -(target_dist * pred_dist).sum(dim=1).mean()

                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()

        # Update Target Network and Log Stats
        epsilon = max(0.01, epsilon * 0.995)

        if episode % 5 == 0:
            agent.target_net.load_state_dict(agent.online_net.state_dict())

        # Write to CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, episode_reward, epsilon])

        if episode % 10 == 0:
            print(f"Episode {episode} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.2f}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_training()
