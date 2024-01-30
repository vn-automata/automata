import base64
import json
import random
from time import sleep
import numpy as np
import bittensor as bt

from automata.protocol import CAsynapse
from automata.validator.reward import get_rewards
from automata.utils.uids import get_random_uids, check_uid_availability


def get_random_params():
    steps = random.randint(10, 20)
    rule_instances = ["Rule30", "Rule110"]
    rule_instance = random.choice(rule_instances)
    bt.logging.info(f"CA parameters: Steps: {steps}, Rule: {rule_instance}")
    return steps, rule_instance


def query_automata_miners(self, steps, rule_instance):
    # TODO: Finetune miner query method.
    responses = []
    miner_uids = []
    try:
        check_uid_availability(
            metagraph=self.metagraph,
            uid=1,
            vpermit_tao_limit=self.config.neuron.vpermit_tao_limit,
        )
        population_size = len(self.metagraph.axons)
        sample_size = min(self.config.neuron.sample_size, population_size)

        if sample_size <= 0:
            bt.logging.error(f"Sample size is zero, cannot query automata miners.")
            return responses, miner_uids  # Return empty responses and miner uids.

        miner_uids = get_random_uids(self, k=sample_size)
        bt.logging.debug(f"Miner uids: {miner_uids}")

        query_responses = self.dendrite.query(
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=CAsynapse(
                steps=steps,
                rule_instance=rule_instance,
            ),
            deserialize=True,
        )

        if query_responses is None:
            raise ValueError("Responses are None, cannot compute scores.")
        else:
            responses = query_responses  # Update only if responses are not None.

    except Exception as e:
        bt.logging.error(f"Failed to query automata miners: {e}")
        sleep(30)  # Wait for 5 seconds before retrying.

    return responses, miner_uids  # Always return an iterable, even if empty.


def compute_scores(self, responses):
    try:
        # TODO: Implement actual scoring & reward logic.
        outputs = [response.payload for response in responses]
        bt.logging.info(f"Received responses: {outputs}")
        scores = [1 for _ in outputs]
        total_score = np.sum(scores)
        if total_score == 0:
            raise ValueError(
                "Total score for responses is zero, cannot compute relative scores."
            )
        scores = np.array(scores) / total_score
        return scores
    except Exception as e:
        bt.logging.error(f"Failed to compute scores: {e}")
        raise


async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    self.sync()
    steps, rule_instance = get_random_params()
    try:
        # Query miners.
        responses, miner_uids = query_automata_miners(self, steps, rule_instance)

        # Check if responses are None
        if responses is None:
            raise ValueError("Responses are None, cannot compute scores.")

        bt.logging.info(f"Responses: {responses}")
        rewards = get_rewards(self, query=steps, responses=responses)
        bt.logging.info(f"Scored responses: {rewards}")

        # Compute scores.
        scores = compute_scores(self, responses)
        bt.logging.info(f"Computed scores: {scores}")

        # Update scores and save state.
        self.update_scores(miner_uids, scores)
        self.save_state()

    except Exception as e:
        bt.logging.error(f"Error in forward process: {e}")
        return None
    return responses
