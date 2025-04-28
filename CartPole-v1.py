import gymnasium as gym
from DQNAgent import Agent
from tqdm import tqdm

env = gym.make("CartPole-v1", render_mode = "rgb_array")

n_episodes = 2000
timesteps = 1000
epsilon = 1.0
epsilon_decay = epsilon / (n_episodes / 1.2)
final_epsilon = 0.01
gamma = 0.85
lr = 1e-4

agent = Agent(lr, env.observation_space.shape[0], env.action_space.n, epsilon, final_epsilon, epsilon_decay, gamma)
rewards = []
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    total_reward = 0
    for t in range(timesteps):
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.push(obs, action, reward, next_obs, terminated)
        if len(agent.replay_buffer) > 100:
            agent.update_main(100)
        if terminated or truncated:
            break
        obs = next_obs
        total_reward += reward
    rewards.append(total_reward)
    agent.decay_epsilon()
    if episode % 5 == 0:
        agent.update_target()
    if episode % 50 == 0:
        print(f"Episode: {episode}, Epsilon: {agent.epsilon:.2f}, Total Reward: {total_reward}")
        env.render()
agent.save_model("cartpole_dqn.pth")
env.close()  

import matplotlib.pyplot as plt


fig, axis = plt.subplots(1, 2, figsize = (20, 10))
axis[0].plot(rewards, label = "Reward")
axis[0].set_xlabel("Episode")
axis[0].set_ylabel("Reward")
axis[1].plot(agent.training_error, label = "Training Error")
axis[1].set_xlabel("Episode")
axis[1].set_ylabel("Training Error")
plt.show()


