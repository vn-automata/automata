from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
import cellpylib as cpl
import bittensor as bt
from abc import ABC, abstractmethod
from protocol import *

# Initialize a 2D cellular automaton with a simple state
initial_state = cpl.init_simple2d(60, 60)

# Create an instance of ConwayRule
rule_instance = ConwayRule()

# Create an instance of Simulate with ConwayRule
sim = Simulate(
    initial_state,
    timesteps=100,
    rule_instance=rule_instance,
    neighbourhood_type="Moore",
    r=1,
)

# Run the simulation
result = sim.run()

# Print the result
print(result)