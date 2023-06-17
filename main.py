from re import T
from environment import FundENV
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import matplotlib.pyplot as plt


env = FundENV()
print(env.action_space.sample())
print(env.observation_space.sample())

'''episodes = 2
for episode in range(1, episodes+1):
    state = env.reset()
    Is_done = False
    score = 0

    while not Is_done:
        action = env.action_space.sample()
        n_state, reward, Is_done, Info = env.step(action)
        score += reward

    print('Episode:{} Score:{}'.format(episode, score))'''



states = env.observation_space.shape
actions = env.action_space.shape[0]


def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + states))
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(BatchNormalization())
    model.add(Dense(24, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(actions, activation='linear'))
    return model


model = build_model(states, actions)
print(model.summary())


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn


dqn = build_agent(model, actions)
dqn.compile(Adam(learning_rate=1e-4), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
