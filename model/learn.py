"""
use the corpus to train and test the model, using the multiprocessing
"""

import time
import multiprocessing
import _queue
from model.matrix import matrix_init, matrix_normal, matrix_dump, matrix_load
from model.counts import read_and_count, read_and_experiment
from common.utils import add_dict, cut_phrase
from common.records import summary_test_result
from common.states import states_space


def train_and_test(set_path: str, train_set: str, test_set: str) -> None:
    """
    the function calls both train and test, train and test the model when called
    """
    train(set_path, train_set)
    print("\n")
    test(set_path, test_set)
    print("\n")


def train(set_path: str, train_set: list) -> None:
    """
    travel through the corpus and do statistics working, namely training the model
    """
    time_start = time.time()
    print(
        "Start training. The train set is:\n",
        ", ".join([cut_phrase(filename)[1] for filename in train_set]),
    )

    # create the group of process
    process_list = []
    queue = multiprocessing.Manager().Queue()
    for filename in train_set:
        process = multiprocessing.Process(
            target=read_and_count, args=(set_path + "/" + filename, queue)
        )
        process.start()
        process_list.append(process)

    # add the process to the main process
    for process in process_list:
        process.join()

    # read the info from queue to matrix
    init_mt, tran_mt, emis_mt = matrix_init()
    while True:
        try:
            result = queue.get(timeout=10)
        except TimeoutError as err:
            print("Timeout occurred:" + str(err))
            break
        except _queue.Empty:
            break

        init_mt = add_dict(init_mt, result["init"])
        tran_mt = add_dict(tran_mt, result["tran"])
        emis_mt = add_dict(emis_mt, result["emis"])

    # calculate the frequency to the probability
    matrix_normal(init_mt, tran_mt, emis_mt)

    # save the matrix to the specified file
    matrix_dump(init_mt, tran_mt, emis_mt)

    time_end = time.time()
    print("Training completed. Time cost:", time_end - time_start, "s")


def test(set_path: str, test_set: str) -> None:
    """
    using the test corpus to evaluate the model's ability.
    """
    time_start = time.time()
    print(
        "Start testing. The test set is:\n",
        ", ".join([cut_phrase(filename)[1] for filename in test_set]),
    )

    # get the existing model matrix previously to reduce IO
    model_mt = matrix_load()

    # create the group of process
    process_list = []
    queue = multiprocessing.Manager().Queue()
    for filename in test_set:
        process = multiprocessing.Process(
            target=read_and_experiment,
            args=(set_path + "/" + filename, model_mt, queue),
        )
        process.start()
        process_list.append(process)

    # add the process to the main process
    for process in process_list:
        process.join()

    # read the info from queue to counts
    rgh_pred, wrg_pred = 0, 0
    err_sample = {state:0 for state in states_space}
    while True:
        try:
            result = queue.get(timeout=10)
        except TimeoutError as err:
            print("Timeout occurred:" + str(err))
            break
        except _queue.Empty:
            break

        rgh_pred += result["rgh"]
        wrg_pred += result["wrg"]
        err_sample = add_dict(err_sample, result["err"])

    # caculate the accuracy and do some recording works
    summary_test_result(rgh_pred, wrg_pred, err_sample)

    time_end = time.time()
    print("Testing completed. Time cost:", time_end - time_start, "s")
