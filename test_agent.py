import gymnasium as gym
import pickle
import torch
from DQNAgent import Agent
from time import sleep
env = gym.make("CartPole-v1", render_mode = "human")
epsilon = 0
n_episodes = 500
epsilon_decay = epsilon / (n_episodes / 2)
final_epsilon = 0.1
gamma = 0.8
lr = 0.01

agent = Agent(lr, env.observation_space.shape[0], env.action_space.n, epsilon, final_epsilon, epsilon_decay, gamma)
agent.load_model("cartpole_dqn.pth")
for episode in range(n_episodes):
    obs, info = env.reset()
    total_reward = 0

    while True:
        action = agent.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        if terminated or truncated:
            break
    print(f"Episode: {episode}, Epsilon: {agent.epsilon:.2f}, Total Reward: {total_reward}")
sleep(3)
env.close()

