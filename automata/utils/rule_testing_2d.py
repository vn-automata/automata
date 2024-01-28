from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
import cellpylib as cpl
import bittensor as bt
from abc import ABC, abstractmethod
from rulesets import *
import subprocess
import matplotlib.pyplot as plt

#initalize 1d
initial_state = cpl.init_simple(100)
#initialize 2d
initial_state = cpl.init_simple2d(100, 100)

# Create an instance of ConwayRule
#rule_instance = Rule30()
rule_instance = ConwayRule()
# Create an instance of Simulate with ConwayRule
sim = Simulate(
    initial_state,
    timesteps=100,
    rule_instance=rule_instance,
    r=1,
)

# Run the simulation
result = sim.run()

# Print the result
print(result)

# Visualize the result
plt.imshow(result[-1], cmap='Greys')
plt.show()
#plt.savefig('`/root/automata1/sim_figs/simulation_result.png')
plt.savefig('`/home/scottrobinson/Downloads/simulation_result.png')
           #subprocess.run(['feh', '/root/automata1/sim_figs/simulation_result.png'])

# Convert the numpy 2D array to ASCII art
#initial_ascii = '\n'.join(''.join('.' if cell else '#' for cell in row) for row in initial_state[-1])
#final_ascii = '\n'.join(''.join('.' if cell else '#' for cell in row) for row in result[-1])

# Convert the numpy 1D array to ASCII artv for all timesteps
final_ascii = '\n'.join(''.join('.' if cell else '#' for cell in row) for row in result)




# Print the ASCII art
print(initial_ascii)
print(final_ascii)