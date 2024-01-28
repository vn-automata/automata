from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
import cellpylib as cpl
import bittensor as bt
from abc import ABC, abstractmethod
from rulesets import *
import subprocess
import matplotlib
matplotlib.use('Qt5Agg')  # 
import matplotlib.pyplot as plt

#initialize 2d
ic = InitialConditions(100, 0.2)
initial_state = ic.init_random_2d(100, 100)
#initial_state = cpl.init_simple2d(60, 60)
#initial_state[:, [28,29,30,30], [30,31,29,31]] = 1

# Create an instance of the rule to be applied
rule_instance = ConwayRule()

# Create an instance of Simulate
sim = Simulate(
    initial_state,
    timesteps=10,
    rule_instance=rule_instance,
    r=1,
)

# Run the simulation
result = sim.run()

# Print the result
print(result)

# Visualize the result
plt.imshow(result[-1], cmap='Blues')
plt.show()



# Convert the numpy 2D array to ASCII art
inital_ascii2D = '\n'.join(''.join('.' if cell == 0 else '#' for cell in row) for state in initial_state for row in state)
print(inital_ascii2D)
#final_ascii = '\n'.join(''.join('.' if cell else '#' for cell in row) for row in result[-1])

# Convert the numpy 2D array to ASCII art for end state
final_ascii = '\n'.join(''.join('.' if cell else '#' for cell in row) for row in result[-1])
print(final_ascii)

# Or use your graphics card!
cpl.plot2d_animate(result)


# Print the ASCII art
print(initial_ascii)
print(final_ascii)