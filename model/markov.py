"""
functions related to Markov chains, including chain generation, backtrack
"""


from common.states import states_space
from common.params import MINIMAL_PROB, ENDSTATE_GUESS, THRESHOULD


def markov_chain_generate(obs_list: str, init_mt: dict, emis_mt: dict) -> list:
    """
    given a list of observations, initial matrix and the emission matrix then return a markov chain
    """
    markov_chain = []
    for obs in obs_list:
        # initial probability of each dicide node is the emission probability
        dicide_node = {
            state: {
                "prev": "",
                # given a very small but not the minmal float as initial probability
                # prevent the all-zero node from masking available information
                "prob": emis_mt[state].get(obs, MINIMAL_PROB)
            }
            for state in states_space
        }
        markov_chain.append(dicide_node)

    # initialize the first node
    for state in states_space:
        markov_chain[0][state]["prob"] *= init_mt[state]

    return markov_chain


def markov_chain_backtrack(markov_chain: list) -> list:
    """
    backtrack from the end state to obtain the state transition route
    """
    # find the end state with the highest probability
    end_state = ENDSTATE_GUESS
    end_prob = markov_chain[-1][ENDSTATE_GUESS]["prob"]
    for state in markov_chain[-1]:
        prob = markov_chain[-1][state]["prob"]
        if prob > end_prob + THRESHOULD:
            end_prob = prob
            end_state = state

    # backtrack from the end state
    transition_route = [end_state]
    for order in range(len(markov_chain) - 1, 0, -1):
        end_state = markov_chain[order][end_state]["prev"]
        transition_route.insert(0, end_state)

    return transition_route
