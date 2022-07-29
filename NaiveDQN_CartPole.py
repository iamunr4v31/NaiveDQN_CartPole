import sys
import pickle
from typing import Iterable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
from typing import Iterable, Union
import gym
import numpy as np
from utils import plot_learning_curve

class Net(nn.Module):
    def __init__(self, lr: float, n_actions: int, input_dims: Union[Iterable[int], int]) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        state = F.relu(self.fc1(state))
        state = self.fc2(state)

        return state

class Agent:
    def __init__(self, input_dims: int, n_actions: int, lr: float, gamma: float=0.99, 
                epsilon: float=1.0, decay_rate: float=1e-5, epsilon_min: float=0.01) -> None:
        self.input_dims = input_dims
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.epsilon_min = epsilon_min
        self.n_actions = n_actions

        self.action_space = list(range(self.n_actions))

        self.Q = Net(lr, n_actions, input_dims)
    
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation, dtype=T.float).to(self.Q.device)
            q_values = self.Q.forward(state)
            action = T.argmax(q_values).item()
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.decay_rate if self.epsilon > self.epsilon_min else self.epsilon_min
    
    def learn(self, state, action, reward, next_state):
        self.Q.optimizer.zero_grad()
        state = T.tensor(state, dtype=T.float).to(self.Q.device)
        action = T.tensor(action).to(self.Q.device)
        reward = T.tensor(reward).to(self.Q.device)
        next_state = T.tensor(next_state, dtype=T.float).to(self.Q.device)

        q_value = self.Q.forward(state)[action]

        q_next = self.Q.forward(next_state).max()

        q_target = reward + self.gamma * q_next

        loss = self.Q.criterion(q_value, q_target).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()
    
    def save_agent(self, filename: str) -> None:
        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    scores, epsilon_history = [], []
    n_games = 10000
    
    agent = Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.n, lr=0.0001)

    for i in range(1, n_games+1):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            agent.learn(observation, action, reward, next_observation)
            observation = next_observation
            score += reward
            # env.render()
        scores.append(score)
        epsilon_history.append(agent.epsilon)
        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            sys.stdout.write(f"\rGame: {i}/{n_games} | Score: {score:.2f} | Average Score: {avg_score:.2f} | Epsilon: {agent.epsilon:.2f}")
            sys.stdout.flush()
    filename = "cartpole_naive_dqn_agent.png"
    x = list(range(1, n_games+1))
    plot_learning_curve(x, scores, epsilon_history, filename)
    agent.save_agent("cartpole_agent.pkl")
    env.close()
