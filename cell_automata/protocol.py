# The MIT License (MIT)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
import cellpylib as cpl
import bittensor as bt
from cell_automata.utils.ca_rules import *


class Simulate:
    """Main simulation runner for CA used in miner and validator routines"""

    def __init__(
        self,
        initial_state: NDArray[Any],
        rule: Callable,
        neighbourhood_type: str = "Moore",
        r: int = 1,
        cell_cycles: int = 100,
        memoize: bool = False,
    ):
        if neighbourhood_type not in ["Moore", "von Neumann"]:
            neighbourhood_type = "Moore"  # default to "Moore" if input is not valid

        self.initial_state = initial_state
        self.rule = rule
        self.neighborhood_type = neighbourhood_type
        self.r = r
        self.cell_cycles = cell_cycles
        self.memoize = memoize

    def run(self) -> NDArray[Any]:
        # Run the cellular automata simulation using cellpylib
        data = cpl.evolve2d(
            cellular_automaton=self.initial_state,
            timesteps=self.cell_cycles,
            apply_rule=self.rule,
            r=self.r,
            neighbourhood=self.neighborhood_type,
            memoize=self.memoize,
        )

        cpl.plot2d_animate(data, cmap="inferno", show_grid=True)
        return data


class ByteTransfer(bt.Synapse):
    """A synapse that verifies the integrity of a simulation"""

    @staticmethod
    def deserialize(bytes: bytes) -> [NDArray[np.float64]]:
        """
        Deserialize the simulation output. This method retrieves the result of
        the CA simulation from the miner in the form of simulation_output,
        deserializes it and returns it as the output of the dendrite.query() call.
        This should be more efficient for numerical ops than a list

        Returns:
        - np.ndarray: The deserialized response, which in this case is the value of simulation_output.
        """
        # Check if the data is not None and deserialize it
        if not isinstance(bytes):
            raise ValueError("Data must be bytes")
        if bytes is not None:
            deserialized = np.frombuffer(bytes, dtype=np.int).reshape(-1, 100)
            data = deserialized.astype(np.float64)

            # Validate the deserialized data (if necessary)

            return data
        return None

    @staticmethod
    def serialize(data: NDArray[np.float64]) -> bytes:
        """
        Serialize the simulation output. This method serializes the result of
        the CA simulation and returns it as the output of the dendrite.query() call.

        Returns:
        - bytes: The serialized response, which in this case is the value of simulation_output.
        """
        # Check if the data is not None and serialize it
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be np.ndarray")
        if data is not None:
            serialized = data.tobytes()

            # Validate the serialized data (if necessary)

        return serialized
