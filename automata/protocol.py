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


import json
import typing
import hashlib
import pydantic
import numpy as np
import bittensor as bt


class IsAlive(bt.Synapse):
    answer: typing.Optional[str] = None
    completion: str = pydantic.Field(
        ...,
        title="Completion",
        description="Completion status of the current request.",
    )


class CAsynapse(bt.Synapse):
    """
    Communication protocol for cellular automata simulation.
    It is a simple request-response protocol where the validator transmits a set of parameters
    to the miner for initialization of a cellular automata simulation. The  miner responds with the
    transformed data upon completion.

    Attributes:
    - array_bytes (bytes): The serialized initial state of the cellular automata sent to the miner.
    - metadata_bytes (bytes): The serialized metadata of the cellular automata parameters sent to the miner.
    - automaton_bytes (bytes): The serialized evolved automaton to return to the validator.
    - automaton_metadata_bytes (bytes): The serialized metadata of the evolved automaton to return to the validator.

    Methods:
    - serialize_parameters: Serialize the initial state and starting parameters for transmission.
    - deserialize_parameters: Deserialize the initial state and starting parameters from transmission.
    - serialize_automaton: Serialize the evolved automaton for transmission.
    - deserialize_automaton: Deserialize the evolved automaton from transmission.
    """

    array_bytes: typing.Optional[bytes] = None
    metadata_bytes: typing.Optional[bytes] = None
    automaton_bytes: typing.Optional[bytes] = None
    automaton_metadata_bytes: typing.Optional[bytes] = None

    def serialize_parameters(
        self,
        initial_state: np.ndarray,
        steps: int,
        rule_func: str,
        neighborhood_func: str,
    ) -> typing.Tuple[bytes, bytes]:
        """
        Serialize the cellular automata configuration and return the bytes for metadata and array.

        Args:
            - initial_state (np.ndarray): The initial state of the automata as a numpy array.
            - steps (int): The number of steps to simulate.
            - rule_func (str): The rule function as a string.
            - neighborhood_func (str): The neighborhood function as a string.

        Returns:
            typing.Tuple[bytes, bytes]: A tuple containing the serialized metadata, array and hash.

        Raises:
            ValueError: If any of the parameters are not in the expected format or type.
        """
        if not isinstance(initial_state, np.ndarray):
            raise ValueError("initial_state must be a numpy array")
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        if not isinstance(rule_func, str) or not rule_func:
            raise ValueError("rule_func must be a non-empty string")
        if not isinstance(neighborhood_func, str) or not neighborhood_func:
            raise ValueError("neighborhood_func must be a non-empty string")

        array_bytes = initial_state.tobytes()
        array_hash = hashlib.sha256(array_bytes).hexdigest()

        metadata = {
            "dtype": str(initial_state.dtype),
            "shape": initial_state.shape,
            "hash": array_hash,
            "steps": steps,
            "rule_func": rule_func,
            "neighborhood_func": neighborhood_func,
        }

        metadata_bytes = json.dumps(metadata).encode("utf-8")
        return metadata_bytes, array_bytes

    def deserialize_parameters(
        metadata_bytes: bytes, array_bytes: bytes
    ) -> typing.Tuple[
        np.ndarray,
        typing.Optional[int],
        typing.Optional[str],
        typing.Optional[str],
    ]:
        """
        Deserialize the parameters and return the cellular automata configuration for running the simulation.

        Args:
            - metadata_bytes (bytes): The serialized metadata of the cellular automata parameters.
            - array_bytes (bytes): The serialized initial state of the cellular automata.

        Returns:
            A tuple containing the configuration parameters for the cellular automata simulation:

            - initial_state (np.ndarray): The initial state of the cellular automata as a numpy array.
            - steps (int): The number of steps to simulate.
            - rule_func (str): The rule function as a string.
            - neighborhood_func (str): The neighborhood function as a string.

        Raises:
            ValueError: Data integrity error if the array hash does not match the hash in the metadata.
        """

        # Deserialize the metadata from bytes to a JSON string and then to a dictionary
        metadata = json.loads(metadata_bytes.decode("utf-8"))

        # Verify the integrity of the array bytes using the hash
        array_hash = hashlib.sha256(array_bytes).hexdigest()
        if array_hash != metadata.get("hash", ""):
            raise ValueError("Data integrity check failed!")

        # Reconstruct the numpy array using the metadata and the array bytes
        initial_state = np.frombuffer(array_bytes, dtype=np.dtype(metadata["dtype"]))
        initial_state = initial_state.reshape(metadata["shape"])

        # Access other metadata if available
        steps = metadata.get("steps")
        rule_func = metadata.get("rule_func")
        neighborhood_func = metadata.get("neighborhood_func")

        # Return the initial state along with any other optional parameters that were provided
        return initial_state, steps, rule_func, neighborhood_func

    def serialize_automaton(self, automaton: np.ndarray) -> bytes:
        """
        Serialize the automaton and return the bytes for the automaton and metadata.

        Args:
            - automaton (np.ndarray): The automaton as a numpy array.

        Returns:
            - automaton_bytes (bytes): The serialized automaton bytes.
            - automaton_metadata_bytes (bytes): The serialized automaton metadata bytes.

        Raises:
            - ValueError: If the automaton is not in the expected format or type.
        """
        if not isinstance(automaton, np.ndarray):
            raise ValueError("automaton must be a numpy array")

        automaton_bytes = automaton.tobytes()
        automaton_hash = hashlib.sha256(automaton_bytes).hexdigest()
        automaton_metadata = {
            "dtype": str(automaton.dtype),
            "shape": automaton.shape,
            "hash": automaton_hash,
        }

        automaton_metadata_bytes = json.dumps(automaton_metadata).encode("utf-8")
        return automaton_bytes, automaton_metadata_bytes

    def deserialize_automaton(
        self, automaton_metadata_bytes: bytes, automaton_bytes: bytes
    ) -> np.ndarray:
        """
        Deserialize the automaton and return the numpy array.

        Args:
            automaton_metadata_bytes (bytes): The serialized metadata of the automaton.
            automaton_bytes (bytes): The serialized automaton.

        Returns:
            np.ndarray: The deserialized automaton as a numpy array.

        Raises:
            ValueError: Data integrity error if the automaton hash does not match the hash in the metadata.
        """

        automaton_metadata = json.loads(automaton_metadata_bytes.decode("utf-8"))
        automaton_hash = hashlib.sha256(automaton_bytes).hexdigest()
        if automaton_hash != automaton_metadata.get("hash", ""):
            raise ValueError("Data integrity check failed!")

        automaton = np.frombuffer(
            automaton_bytes, dtype=np.dtype(automaton_metadata["dtype"])
        )
        automaton = automaton.reshape(automaton_metadata["shape"])

        return automaton

    def __str__(self):
        return (
            f"CAsynapse(array_bytes={self.array_bytes[:12]}, "
            f"metadata_bytes={self.metadata_bytes[:12]}, "
            f"automaton_bytes={self.automaton_bytes[:12]}, "
            f"automaton_metadata_bytes={self.automaton_metadata_bytes[:12]}, "
            f"axon={self.axon.dict()}, "
            f"dendrite={self.dendrite.dict()}"
        )
  