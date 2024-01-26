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

import os
import time
import random

import numpy as np
import bittensor as bt
import cellpylib as cpl

import automata
from automata.utils import rulesets
from automata.protocol import CAsynapse
from automata.utils.uids import get_random_uids
from automata.validator import forward
from automata.validator.reward import get_rewards, reward
from automata.validator.validator import BaseValidatorNeuron


class Validator(BaseValidatorNeuron):
    
    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

    def get_random_params(self):
        # Generate a random initial state as a 2D numpy array
        initial_state = np.random.randint(2, size=(10, 10))

        # Choose a random number of steps
        steps = random.randint(50, 100)

        # Choose a random rule function. There should be a better way than adding new strings each time!
        rule_funcs = [
                "Conway",
                "HighLife",
                "DayAndNight",
            ]
        rule_func = random.choice(rule_funcs)

        # Choose a random neighborhood function. There should be a better way than adding new strings each time!
        neighborhood_funcs = ["Moore", "Von Neumann"]
        neighborhood_func = random.choice(neighborhood_funcs)
            
        # Log and return the parameters.
        if initial_state is not None and steps is not None and rule_func is not None and neighborhood_func is not None:
            bt.logging.info(
                f"Generated cellular automata parameters: {initial_state}, {steps}, {rule_func}, {neighborhood_func}"
            )
        return initial_state, steps, rule_func, neighborhood_func
    
    
    def query_automata_miners(self, initial_state, steps, rule_func, neighborhood_func):
        miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)
        responses = self.dendrite.query(
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=CAsynapse(
                initial_state,
                steps, 
                rule_func, 
                neighborhood_func),
            deserialize=True,
        )
        return responses, miner_uids
    
    
    def compute_scores(self, responses):
        outputs = [response.payload for response in responses]
        bt.logging.info(f"Received responses: {outputs}")
        scores = [np.sum(output) for output in outputs]
        scores = np.array(scores) / np.sum(scores)
        return scores

    async def forward(self):
        """
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
                # Sync with the metagraph and get the miner uids.
        self.sync()
           
        # Get the params for the CA simulation.
        initial_state, steps, rule_func, neighborhood_func = self.get_random_params()
        bt.logging.info(f"Params: {initial_state}, {steps}, {rule_func}, {neighborhood_func}")
        
        # Query the network for the CA simulation results.
        responses = self.query_network(initial_state, steps, rule_func, neighborhood_func)
        miner_uids = []
        bt.logging.info(f"Responses: {responses}")
        
        # Get the rewards for the responses.
        rewards = get_rewards(self, query=self.step, responses=responses)
        bt.logging.info(f"Scored responses: {rewards}")
        
        # Update the scores based on the rewards.
        self.compute_scores(rewards)
        self.update_scores(rewards, miner_uids)
        
        # Save the state.
        self.save_state()
        

        
        
        
        return await forward(self)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(5)
