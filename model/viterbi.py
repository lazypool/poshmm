"""
implemented the viterbi algorithm, used to decoder the hidden state
"""


from model.markov import (
    markov_chain_generate,
    markov_chain_backtrack,
)
from common.params import MINIMAL_PROB, MIDSTATE_GUESS


def state_decoder(obs_list: list, init_mt: dict, tran_mt: dict, emis_mt: dict) -> list:
    """
    given a list of observations and the 3 matrix, decode the list's hidden state
    """
    # generate an initialized markov-chain
    markov_chain = markov_chain_generate(obs_list, init_mt, emis_mt)

    # update it with viterbi algorithm
    viterbi_decode(markov_chain, tran_mt)

    # obtain the most likely state transition route
    transition_route = markov_chain_backtrack(markov_chain)

    return transition_route


def viterbi_decode(markov_chain: list, tran_mt: dict) -> None:
    """
    update the markov chain by Viterbi algorithm
    """
    # iterate to the end node from the second node
    for order in range(1, len(markov_chain)):
        # get the previous and current decide node
        prev_node = markov_chain[order - 1]
        curr_node = markov_chain[order]

        # for each current state find argmax previous state and get max probability
        for curr_state in curr_node:
            max_state, max_prob = "", 0

            # travel through the previous node to find the argmax state
            for prev_state in prev_node:
                prob = (
                    # the previous's probability
                    prev_node[prev_state]["prob"]
                    # the transition probability
                    * tran_mt[prev_state][curr_state]
                    # the emission probability
                    * curr_node[curr_state]["prob"]
                )
                if prob >= max_prob:
                    max_state = prev_state
                    max_prob = prob

            # prevent the all-zero situation
            if max_prob == 0.0:
                max_state = MIDSTATE_GUESS
                max_prob = MINIMAL_PROB

            # update the current node
            curr_node[curr_state]["prev"] = max_state
            curr_node[curr_state]["prob"] = max_prob
