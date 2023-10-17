"""
implement the function doing the predicting work
"""


from model.matrix import matrix_load
from model.viterbi import state_decoder


def predict(origin_sent: str, cut_func: any) -> str:
    """
    given an original sentence, and then return a tagged sentence
    """
    # cut the sentence to words
    word_list = cut_func(origin_sent.strip())

    # load the matrix from save
    init_mt, tran_mt, emis_mt = matrix_load()

    # decode the word to its hidden state
    state_list = state_decoder(word_list, init_mt, tran_mt, emis_mt)

    # concatenate the word and its state then return
    tagged_list = [word + "/" + state for word, state in zip(word_list, state_list)]
    return " ".join(tagged_list)
