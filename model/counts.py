"""
reads the text and counts the matirx, used for the multiprocessing
"""


from common.states import states_space
from common.utils import cut_phrase
from model.matrix import matrix_init
from model.viterbi import state_decoder


def read_and_count(file_path: str, shr_queue: any) -> None:
    """
    given the absolute path, reads and counts, and then save it to the 3 dict given
    """
    # generate the 3 matrix with default 0 valuee
    init_mt, tran_mt, emis_mt = matrix_init()

    # read the text and count it
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # get the list of the tuple (word, state)
            word_pairs = [cut_phrase(phrase) for phrase in line.strip().split(" ")]

            # travel through the phrase list and do calculate
            length = len(word_pairs)
            for order in range(0, length):
                # skip if curretn state is illegal
                current = word_pairs[order][1]
                if current not in states_space:
                    continue

                # update the emission matrix
                word = word_pairs[order][0]
                try:
                    emis_mt[current][word] += 1
                except KeyError:
                    emis_mt[current][word] = 1

                # update the initial matrix
                if order == 0:
                    init_mt[current] += 1
                    continue

                # update the transition matrix
                before = word_pairs[order - 1][1]
                if before not in states_space:
                    continue
                tran_mt[before][current] += 1

    # put the result into the shr_queue
    try:
        shr_queue.put({"init": init_mt, "tran": tran_mt, "emis": emis_mt})
    except AttributeError:
        return


def read_and_experiment(file_path: str, model_mt: dict, shr_queue: any) -> None:
    """
    read the given file and use the file's content to do experiment on existing model
    """
    # the right predicate and the wrong predicate
    rgh_pred, wrg_pred = 0, 0
    err_sample = {state:0 for state in states_space}

    # read the text and use it
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # get the observed values and target values
            obs_list, tar_list = [], []
            for word, state in [cut_phrase(phrase) for phrase in line.strip().split(" ")]:
                if state not in states_space:
                    continue
                if word != "" and state != "":
                    obs_list.append(word)
                    tar_list.append(state)

            # conduct the predict func
            pred_list = state_decoder(obs_list, model_mt[0], model_mt[1], model_mt[2])

            # compare the predict result and ground truth result
            assert len(pred_list) == len(tar_list), \
                "The predicted result does not match the actual result length."
            for pred, tar in zip(pred_list, tar_list):
                if pred == tar:
                    rgh_pred += 1
                else:
                    err_sample[tar] += 1
                    wrg_pred += 1

    # put the result into the shr_queue
    try:
        shr_queue.put({"rgh": rgh_pred, "wrg": wrg_pred, "err": err_sample})
    except AttributeError:
        return
