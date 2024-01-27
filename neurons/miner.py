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

import time
import typing
import numpy as np
import bittensor as bt
import cellpylib as cpl

import automata
from automata.utils import rulesets
from automata.miner.miner import BaseMinerNeuron


# Cellular automata initial state functions for testing.
function_mapping = {
    "cpl.init_simple": cpl.init_simple,
    "cpl.init_simple2d": cpl.init_simple2d,
    "cpl.init_random": cpl.init_random,
    "cpl.init_random2d": cpl.init_random2d,
}

# Cellular automata rule functions.
rule_class_mapping = {
    "Conway": rulesets.ConwayRule,
    "HighLife": rulesets.HighLifeRule,
    "DayAndNight": rulesets.DayAndNightRule,
}


class miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super(miner, self).__init__(config=config)

    def evolve_automata(
        self,
        initial_state: np.ndarray,
        steps: int,
        rule_instance: rulesets.Rule,
        r: int,
        neighbourhood_func: str,
        memoize: str or bool or None,
    ) -> np.ndarray:
        """
        Simulate a cellular automata with the given parameters.

        Args:
            initial_state (NDArray): The initial state of the cellular automata.
            timesteps (int): The number of timesteps to simulate.
            rule_instance (rulesets.Rule): The rule to apply to the cellular automata.
            r (int): The radius of the neighbourhood.
            neighbourhood_type (str): The type of neighbourhood to use.
            memoize (str): The memoization type to use.

        Returns:
            NDArray: The evolved state of the cellular automata.
        """
        bt.logging.trace(f"Simulating cellular automata with {steps} timesteps.")

        automaton = cpl.evolve2d(
            cellular_automaton=initial_state,
            timesteps=steps,
            apply_rule=rule_instance.apply_rule,
            r=r,
            neighbourhood=neighbourhood_func,
            memoize=memoize,
        )
        return automaton

    async def forward(
        self, synapse: automata.protocol.CAsynapse
    ) -> automata.protocol.CAsynapse:
        # Receive simulation parameters from the validator.
        (
            initial_state,
            steps,
            neighbourhood_func,
            rule_func,
        ) = synapse.deserialize_parameters()

        # Log the parameters.
        bt.logging.info(
            f"Received cellular automata request from {synapse.dendrite.hotkey}."
        )
        bt.logging.info(f"Inital state: {initial_state}")
        bt.logging.info(f"Timesteps: {steps}")
        bt.logging.info(f"Neighbourhood type: {neighbourhood_func}")
        bt.logging.info(f"Rule function: {rule_func}")

        # Map rule_func str to callable rule class dict.
        rule_class_name = rule_func
        rule_class = rule_class_mapping.get(rule_class_name, None)
        if rule_class is None:
            raise ValueError(f"Rule {rule_class_name} is not recognized.")
        rule_instance = rule_class()

        # Generate the cellular automata.
        automaton = self.evolve_automata(
            initial_state=initial_state,
            timesteps=steps,
            rule_instance=rule_instance.apply_rule,
            r=1,
            neighbourhood_type=neighbourhood_func,
            memoize=None,
        )
        if automaton is None:
            raise ValueError("Automaton could not be generated.")
        else:
            bt.logging.info(
                f"Simulated cellular automata over {steps} timesteps with {rule_class_name}."
            )

        # Return the response to the validator.
        automaton = automaton.tobytes()
        synapse.automaton_bytes = automaton
        bt.logging.info(f"Transmitting serialized automaton to {synapse.dendrite.hotkey}.")
        return synapse

    async def blacklist(
        self, synapse: automata.protocol.CAsynapse
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.Dummy): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        # TODO(developer): Define how miners should blacklist requests.
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from unrecognized entities.
            bt.logging.trace(
                f"Blacklisting unrecognized hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: automata.protocol.CAsynapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (bt.Synapse): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority
        )
        return prirority


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with miner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)
