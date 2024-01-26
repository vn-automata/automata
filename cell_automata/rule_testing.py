from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
import cellpylib as cpl
import bittensor as bt
from abc import ABC, abstractmethod
from protocol import *
import subprocess



initial_state = cpl.init_simple(100)

# Create an instance of ConwayRule
rule_instance = Rule30()

# Create an instance of Simulate with ConwayRule
sim = Simulate1D(
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
#plt.imshow(result[-1], cmap='Greys')
#plt.savefig('/root/automata1/sim_figs/simulation_result.png')

#subprocess.run(['feh', '/root/automata1/sim_figs/simulation_result.png'])

# Convert the numpy array to ASCII art
initial_ascii = '\n'.join(''.join('.' if cell else '#' for cell in row) for row in initial_state[-1])
final_ascii = '\n'.join(''.join('.' if cell else '#' for cell in row) for row in result[-1])

# Print the ASCII art
print(initial_ascii)
print(final_ascii)