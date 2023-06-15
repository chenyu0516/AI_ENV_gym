from environment import FundENV

env = FundENV()
print(env.action_space.sample())
print(env.observation_space.sample())