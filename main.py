from environment import FundENV
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


env = FundENV()
print(env.action_space.sample())
print(env.observation_space.sample())

episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    Is_done = False
    score = 0

    while not Is_done:
        action = env.action_space.sample()
        current_value, n_state, reward, info, timer, Is_Done, df = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))

'''
states = env.observation_space.shape
actions = env.action_space.n


def build_model(states, actions):
    model = Sequential()
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
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
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)'''