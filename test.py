from gym.spaces import Discrete, Box, Dict, Sequence, Tuple
import numpy as np

n = 4  # Number of elements in the tuple space

list_1 = [Box(low=-1, high=1, shape=(1,), dtype=np.float32), Box(low=0, high=float('inf'),
                                                                         shape=(1,), dtype=np.float32)]
# there are n traders
list_2 = [Box(low=-float('inf'), high=float('inf'), shape=(1,), dtype=np.float32) for _ in range(n)]
element_space = list_1+list_2
observation_space = Tuple(element_space)
print(observation_space.sample())
print((0, 10000, [0]*n))