from gc import callbacks
from re import T
from environment import FundENV
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras import backend as K
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import matplotlib.pyplot as plt
import numpy as np


env = FundENV()
print(env.action_space.sample())
print(env.observation_space.sample())



states = env.observation_space.shape
actions = env.action_space.shape[0]


def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + states))
    model.add(Dense(96, activation='relu', input_shape=states))
    model.add(BatchNormalization())
    model.add(Dense(96, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(actions, activation='linear'))
    return model


model = build_model(states, actions)
print(model.summary())


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=2000, target_model_update=1e-2)
    return dqn


dqn = build_agent(model, actions)
dqn.compile(Adam(learning_rate=1e-4), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
dqn.compile(Adam(learning_rate=1e-3, clipvalue=1.0), metrics=['mae'])


dqn.save_weights("D:\\python projects\\AI_env\\dqn_weights.h5f")

dqn.load_weights('dqn_weights.h5f')


def predict_action(state):
    state = np.expand_dims(state, axis=0)  # reshape (1, 10) to (1, 1, 10)
    q_values = dqn.compute_q_values(state)
    action = np.argmax(q_values[0])
    return action


episodes = 1
for episode in range(1, episodes+1):
    state = env.reset()
    Is_done = False
    score = 0

    while not Is_done:
        action = predict_action(state)
        state, reward, Is_done, info = env.step(action)
        current_value, timer, df = env.value, env.timer, env.df
        score += reward

    print(df)
    df['SCT'].plot()
    plt.show()
    print('Episode:{} Score:{}'.format(episode, score))