# test_single_step.py
import unittest
from neurons.validator import Neuron as Validator
from neurons.miner import Neuron as Miner
from automata.protocol import Dummy
from automata.utils.uids import get_random_uids
from automata.validator.validator import BaseValidatorNeuron
from bittensor import config as btconfig

class SingleStepInteractionTestCase(unittest.TestCase):
    def setUp(self):
        # Validator setup
        validator_config = BaseValidatorNeuron.config()
        validator_config.wallet._mock = True
        validator_config.metagraph._mock = True
        validator_config.subtensor._mock = True
        self.validator = Validator(validator_config)

        # Miner setup
        miner_config = btconfig()  # Assuming a similar config method exists for the miner
        miner_config.wallet._mock = True
        miner_config.metagraph._mock = True
        miner_config.subtensor._mock = True
        self.miner = Miner(miner_config)

        # Generate mock UIDs for miners
        self.miner_uids = get_random_uids(self, k=10)

    def test_run_single_step(self):
        print("Starting single step interaction test...")

        # Simulate a query from the validator to the miner
        print("Querying the miner...")
        responses = self.validator.dendrite.query(
            axons=[self.miner.axon_info for _ in self.miner_uids],
            synapse=Dummy(dummy_input="test_input"),
            deserialize=True,
        )

        print("Checking responses...")
        # Check the response
        for response in responses:
            self.assertIsNotNone(response)  # Ensure we got a response
            print(f"Received response: {response}")

        # Example of printing additional information
        print("Ruleset chosen: ExampleRuleset")
        print("Neighborhood: ExampleNeighborhood")

        print("Single step interaction test completed.")

if __name__ == "__main__":
    unittest.main()