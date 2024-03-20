import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define model
def build_dqn_model(in_states, h1_nodes, out_actions):
    model = Sequential()
    model.add(Dense(h1_nodes, input_shape=(in_states,), activation='relu'))
    model.add(Dense(out_actions, activation='linear'))
    return model

# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# FrozenLake Deep Q-Learning
class FrozenLakeDQL():
    def __init__(self):
        self.learning_rate_a = 0.001
        self.discount_factor_g = 0.9    
        self.network_sync_rate = 10
        self.replay_memory_size = 1000
        self.mini_batch_size = 32
        self.ACTIONS = ['L','D','R','U']

    # Converts a state (int) to a one-hot encoded vector.
    def state_to_dqn_input(self, state, num_states):
        input_tensor = np.zeros((1, num_states))
        input_tensor[0, state] = 1
        return input_tensor

    # Train the FrozenLake environment
    def train(self, episodes, render=False, is_slippery=False):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human' if render else None)
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        
        epsilon = 1.0
        memory = ReplayMemory(self.replay_memory_size)

        policy_dqn = build_dqn_model(num_states, num_states, num_actions)
        target_dqn = build_dqn_model(num_states, num_states, num_actions)
        policy_dqn.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate_a))

        print('Policy (random, before training):')
        self.print_dqn(policy_dqn, num_states)

        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []
        step_count = 0

        for i in range(episodes):
            state = env.reset()[0]
            terminated = False

            while not terminated:
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(policy_dqn.predict(self.state_to_dqn_input(state, num_states)))

                new_state, reward, terminated, _ = env.step(action)
                memory.append((state, action, new_state, reward, terminated))
                state = new_state
                step_count += 1

                if reward == 1:
                    rewards_per_episode[i] = 1

                if len(memory) > self.mini_batch_size:
                    self.optimize(memory.sample(self.mini_batch_size), policy_dqn, target_dqn, num_states)

                epsilon = max(epsilon - 1/episodes, 0.01)
                epsilon_history.append(epsilon)

                if step_count >= self.network_sync_rate:
                    target_dqn.set_weights(policy_dqn.get_weights())
                    step_count = 0

        env.close()
        policy_dqn.save('frozen_lake_dql_tf.h5')

        self.plot_results(rewards_per_episode, epsilon_history)

    def optimize(self, mini_batch, policy_dqn, target_dqn, num_states):
        states = np.zeros((self.mini_batch_size, num_states))
        next_states = np.zeros((self.mini_batch_size, num_states))
        actions, rewards, dones = [], [], []

        for i, (state, action, next_state, reward, done) in enumerate(mini_batch):
            states[i] = self.state_to_dqn_input(state, num_states)
            next_states[i] = self.state_to_dqn_input(next_state, num_states)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        q_values = policy_dqn.predict(states)
        q_values_next = target_dqn.predict(next_states)

        for i in range(self.mini_batch_size):
            if dones[i]:
                q_values[i, actions[i]] = rewards[i]
            else:
                q_values[i, actions[i]] = rewards[i] + self.discount_factor_g * np.max(q_values_next[i])

        policy_dqn.fit(states, q_values, epochs=1, verbose=0)

    def print_dqn(self, dqn, num_states):
        for s in range(num_states):
            q_values = dqn.predict(self.state_to_dqn_input(s, num_states))[0]
            best_action = self.ACTIONS[np.argmax(q_values)]
            print(f'{s:02},{best_action},[{", ".join(f"{qv:+.2f}" for qv in q_values)}]', end=' ')
            if (s + 1) % 4 == 0:
                print()

    def plot_results(self, rewards_per_episode, epsilon_history):
        plt.figure(figsize=(12, 5))

        plt.subplot(121)
        plt.plot(np.cumsum(rewards_per_episode), label='Cumulative Rewards')
        plt.title('Cumulative Rewards per Episode')
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Rewards')
        plt.legend()

        plt.subplot(122)
        plt.plot(epsilon_history, label='Epsilon')
        plt.title('Epsilon Decay')
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        plt.legend()

        plt.savefig('frozen_lake_dql_tf.png')
        plt.show()

    # Run the FrozenLake environment with the learned policy
    def test(self, episodes, is_slippery=False):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human')
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        policy_dqn = build_dqn_model(num_states, num_states, num_actions)
        policy_dqn.load_weights('frozen_lake_dql_tf.h5')

        print('Policy (trained):')
        self.print_dqn(policy_dqn, num_states)

        for _ in range(episodes):
            state = env.reset()[0]
            terminated = False

            while not terminated:
                action = np.argmax(policy_dqn.predict(self.state_to_dqn_input(state, num_states)))
                state, _, terminated, _ = env.step(action)

        env.close()

if __name__ == '__main__':
    frozen_lake = FrozenLakeDQL()
    frozen_lake.train(1000)
    frozen_lake.test(10)
