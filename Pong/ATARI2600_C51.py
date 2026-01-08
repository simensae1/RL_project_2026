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

# --- Network Definitions (Same as before) ---


class CategoricalDQN(nn.Module):
    def __init__(self, input_shape, num_actions, num_atoms=51, Vmin=-10, Vmax=10):
        super(CategoricalDQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.register_buffer("support", torch.linspace(Vmin, Vmax, num_atoms))  #

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)  # 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        fc_input_dim = self.feature_size(input_shape)
        self.fc1 = nn.Linear(fc_input_dim, 512)
        self.fc_q = nn.Linear(512, num_actions * num_atoms)  #Instead of having only one output, the network outputs 51*num_actions.  

    def feature_size(self, input_shape):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *input_shape)))).view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        q_logits = self.fc_q(x).view(-1, self.num_actions, self.num_atoms)  #this reshape the output to match a distribution of the 51 atoms for each possible action 
        return F.log_softmax(q_logits, dim=2)  

    def get_q_value(self, x):
        log_probs = self.forward(x)
        probs = log_probs.exp()
        return (probs * self.support).sum(dim=2)


def projection_distribution(next_dist, rewards, dones, gamma, num_atoms, Vmin, Vmax, support):  #algorithm 1 of the article
    batch_size = rewards.size(0)
    delta_z = (Vmax - Vmin) / (num_atoms - 1)
    Tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * gamma * support.unsqueeze(0)
    Tz = Tz.clamp(min=Vmin, max=Vmax)  # clamp is the operator [.]_{Vmin}^{Vmax}
    b = (Tz - Vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    m = torch.zeros(batch_size, num_atoms).to(rewards.device)
    offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long().unsqueeze(1).expand(batch_size, num_atoms).to(rewards.device)

    m.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    m.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
    return m

# --- Agent Class ---


class C51Agent:
    def __init__(self, state_shape, action_size, num_atoms, device="cuda"):
        self.device = device
        self.action_size = action_size
        self.num_atoms = num_atoms
        self.Vmin = -10
        self.Vmax = 10
        self.batch_size = 32
        self.gamma = 0.99
        self.learning_rate = 0.00025
        self.update_target_freq = 10000

        self.online_net = CategoricalDQN(state_shape, action_size, self.num_atoms, self.Vmin, self.Vmax).to(device)
        self.target_net = CategoricalDQN(state_shape, action_size, self.num_atoms, self.Vmin, self.Vmax).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.learning_rate, eps=0.01 / self.batch_size)
        self.steps = 0

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net.get_q_value(state_t)
            return q_values.argmax(dim=1).item()

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        with torch.no_grad():
            next_q_values = self.target_net.get_q_value(next_states)
            next_actions = next_q_values.argmax(1)
            next_log_probs = self.target_net(next_states)
            next_probs = next_log_probs.exp()
            next_best_dist = next_probs[range(len(next_actions)), next_actions]

            target_dist = projection_distribution(
                next_best_dist, rewards, dones, self.gamma,
                self.num_atoms, self.Vmin, self.Vmax, self.online_net.support
            )

        log_probs = self.online_net(states)
        pred_dist = log_probs[range(len(actions)), actions]

        loss = - (target_dist * pred_dist).sum(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.update_target_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return loss.item()


class ReplayBuffer:
    def __init__(self, capacity=100000):  # Reduced capacity for RAM safety in example
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


def run_training(game_id, num_atoms, seed, total_frames=11_000_000):
    # Setup
    env = gym.make(game_id, render_mode=None, frameskip=1)
    env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4, scale_obs=False)
    env = TransformReward(env, lambda r: np.sign(r))
    env = FrameStackObservation(env, stack_size=4)

    # Seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = C51Agent((4, 84, 84), env.action_space.n, num_atoms, device)
    print("-" * 30)
    print(f"DEBUG: Atom values (Support) for {num_atoms} atoms:")
    print(agent.online_net.support)
    print("-" * 30)
    buffer = ReplayBuffer(capacity=1000000)  # Full buffer size for reproduction

    # Logging
    log_filename = f"results_{game_id.split('/')[-1]}_atoms{num_atoms}_seed{seed}_200M.csv"
    with open(log_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'score', 'loss'])

    # Epsilon Schedule for Figure 3
    # "For this experiment, we set epsilon=0.05"
    epsilon = 0.05

    state, _ = env.reset(seed=seed)
    state = np.array(state)
    episode_reward = 0
    recent_loss = 0

    print(f"Starting training on {game_id} with {num_atoms} atoms. Device: {device}")

    for frame_idx in range(1, total_frames + 1):
        # Action
        action = agent.select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = np.array(next_state)

        buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        # Training
        if len(buffer) > 1000 and frame_idx % 4 == 0: #not all frame are used for training 
            states, actions, rewards, next_states, dones = buffer.sample(32)  #
            recent_loss = agent.train_step(states, actions, rewards, next_states, dones)

        # Episode End
        if done:
            # Log Result
            with open(log_filename, 'a', newline='') as f:
                csv.writer(f).writerow([frame_idx, episode_reward, recent_loss])

            print(f"Frame: {frame_idx}, Score: {episode_reward}, Loss: {recent_loss:.4f}")
            episode_reward = 0
            state, _ = env.reset()
            state = np.array(state)


if __name__ == "__main__":
    liste_atomes = [2]
    for atomes in liste_atomes:
        parser = argparse.ArgumentParser()
        parser.add_argument("--game", type=str, default="ALE/Pong-v5", help="Gymnasium ID")
        parser.add_argument("--atoms", type=int, default=atomes, help="Number of atoms (5, 11, 21, 51)")
        parser.add_argument("--frames", type=int, default=11_000_000, help="Total training frames")
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        args = parser.parse_args()

        run_training(args.game, args.atoms, args.seed, args.frames)
