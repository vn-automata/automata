from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
import cellpylib as cpl
import bittensor as bt
from abc import ABC, abstractmethod
from protocol import *
import subprocess



# Initialize a 2D cellular automaton with a simple state
initial_state = cpl.init_simple2d(60, 60)

# Create an instance of ConwayRule
rule_instance = ConwayRule()

# Create an instance of Simulate with ConwayRule
sim = Simulate(
    initial_state,
    timesteps=5,
    rule_instance=rule_instance,
    neighbourhood_type="Moore",
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