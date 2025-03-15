import tensorflow as tf
import numpy as np
import numpy.random as random
from keras.api.layers import Dense
from keras.api.models import Sequential
from keras.api.optimizers import Adam
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import deque

env = gym.make("CliffWalking-v0", render_mode = "rgb_array", is_slippery = True)
state_size = env.observation_space.n
action_size = env.action_space.n

#Tabular Q-Learning agent
class TabularQLearning:
    def __init__(self, state_size, action_size, learning_rate, discount_rate, epsilon = 1.0, epsilon_decay = 0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.lr = learning_rate
        self.dr = discount_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
    
    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.dr * next_max)
        self.q_table[state, action] = new_value
        self.epsilon *= self.epsilon_decay

# DQN implementation
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.dr = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.lr = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        indices = random.choice(len(self.memory), batch_size, replace=False)
        minibatch = [self.memory[i] for i in indices]
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        targets = rewards + self.dr * (np.amax(self.model.predict(next_states, verbose=0), axis=1)) * (1 - dones)
        targets_full = self.model.predict(states, verbose=0)
        ind = np.array([i for i in range(batch_size)])
        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training function
def train_agent(agent, episodes, is_dqn=False):
    rewards_history = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            
            if is_dqn:
                agent.remember(state, action, reward, next_state, done)
                agent.replay(32)
            else:
                agent.update(state, action, reward, next_state)
                
            state = next_state
            
        rewards_history.append(total_reward)
        print(f"Episode: {episode}, Reward: {total_reward}")
        
        if episode % 100 == 0:
            print(f"Episode: {episode}, Average Reward: {np.mean(rewards_history[-100:])}")
    
    return rewards_history

tabular_agent = TabularQLearning(state_size, action_size, learning_rate = 0.01, discount_rate = 0.99)
tabular_rewards = train_agent(tabular_agent, episodes = 1000)

# Training DQN
#dqn_agent = DQNAgent(state_size, action_size)
#dqn_rewards = train_agent(dqn_agent, episodes=1000, is_dqn=True)


# Plot results
plt.figure(figsize=(10, 5))
plt.plot(tabular_rewards, label='Tabular Q-Learning')
#plt.plot(dqn_rewards, label='DQN')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Performance Comparison')
plt.legend()
plt.show()