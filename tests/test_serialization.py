import unittest
import numpy as np
from automata.protocol import CAsynapse

class TestSingleStepInteraction(unittest.TestCase):
    def test_single_step_interaction(self):
        # Create an instance of the CAsynapse class
        ca_synapse = CAsynapse()

        # Define test parameters
        initial_state = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        steps = 10
        rule_func = "Conway's Game of Life"
        neighborhood_func = "Moore Neighborhood"

        # Print the parameters
        print("Initial State:")
        print(initial_state)
        print("Steps:", steps)
        print("Rule Function:", rule_func)
        print("Neighborhood Function:", neighborhood_func)

        # Serialize the parameters
        metadata_bytes, array_bytes = ca_synapse.serialize_parameters(initial_state, steps, rule_func, neighborhood_func)

        # Print the serialized metadata and array
        print("Serialized Metadata Bytes:", metadata_bytes)
        print("Serialized Array Bytes:", array_bytes)

        # Deserialize the parameters
        deserialized_params = ca_synapse.deserialize_parameters(metadata_bytes, array_bytes)

        # Print the deserialized parameters
        print("Deserialized Parameters:")
        print("Initial State:")
        print(deserialized_params[0])
        print("Steps:", deserialized_params[1])
        print("Rule Function:", deserialized_params[2])
        print("Neighborhood Function:", deserialized_params[3])

if __name__ == '__main__':
    unittest.main()