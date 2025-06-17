import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import config

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=config.REPLAY_BUFFER_SIZE)
        self.gamma = config.DISCOUNT_FACTOR    # discount rate
        self.epsilon = config.EPSILON_START  # exploration rate
        self.epsilon_min = config.EPSILON_END
        self.epsilon_decay_steps = config.EPSILON_DECAY_STEPS
        self.learning_rate = config.LEARNING_RATE
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.train_step_counter = 0

    def _build_model(self):
        # Simple MLP for Q-value approximation
        model = keras.Sequential([
            keras.layers.Input(shape=(self.state_size,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear') # Q-values per action
        ])
        model.compile(loss='mse',
                      optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Explore
        state = np.reshape(state, [1, self.state_size]) # Add batch dimension
        act_values = self.model.predict(state, verbose=0) # Exploit
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0 # Not enough samples yet

        minibatch = random.sample(self.memory, batch_size)
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        # Predict Q-values for starting states using the main model
        q_values_current = self.model.predict(states, verbose=0)

        # Predict target Q-values fromèbrext states using the target model
        q_values_next_target = self.target_model.predict(next_states, verbose=0)

        # Calculate target Q-values using the Bellman equation
        # target = reward + gamma * max(Q(next_state))  (for non-terminal states)
        # target = reward (for terminal states)
        targets = rewards + self.gamma * np.max(q_values_next_target, axis=1) * (1 - dones)

        # Update the Q-value for the action actually taken
        target_q_values = q_values_current
        batch_indices = np.arange(batch_size)
        target_q_values[batch_indices, actions] = targets

        # Train the main model
        history = self.model.fit(states, target_q_values, epochs=1, verbose=0)
        loss = history.history['loss'][0]

        # Update target network periodically
        self.train_step_counter += 1
        if self.train_step_counter % config.TARGET_UPDATE_FREQ == 0:
            self.update_target_model()
            # print("Target network updated")

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
             # Linear decay example
             self.epsilon -= (config.EPSILON_START - config.EPSILON_END) / self.epsilon_decay_steps
             self.epsilon = max(self.epsilon_min, self.epsilon)


        return loss

    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model() # Ensure target model is also updated

    def save(self, name):
        self.model.save_weights(name)