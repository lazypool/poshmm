"""
the method relation to the matrix, shared by all programs
"""


import os
import json
from common.states import states_space
from common.utils import normalize_dict

SAVE_PATH = os.path.abspath(".")
init_sv = SAVE_PATH + "/matrix/initial.json"
tran_sv = SAVE_PATH + "/matrix/transition.json"
emis_sv = SAVE_PATH + "/matrix/emission.json"


def matrix_init() -> (dict, dict, dict):
    """
    generate the initial matrix: init_mt, tran_mt, emis_mt
    """
    init_mt, tran_mt, emis_mt = {}, {}, {}
    matrix_clean(init_mt, tran_mt, emis_mt)
    return init_mt, tran_mt, emis_mt


def matrix_clean(init_mt: dict, tran_mt: dict, emis_mt: dict) -> None:
    """
    clean the 3 matrix to the default 0 value
    """
    # clean the initial matrix
    init_mt.clear()
    for state in states_space:
        init_mt[state] = 0

    # clean the transition matrix
    tran_mt.clear()
    for state in states_space:
        tran_mt[state] = {key: 0 for key in states_space}

    # clean the emission matrix
    emis_mt.clear()
    for state in states_space:
        emis_mt[state] = {}


def matrix_dump(init_mt: dict, tran_mt: dict, emis_mt: dict) -> None:
    """
    save the 3 matrix to the specified file
    """
    # save the initial matrix
    with open(init_sv, "w", encoding="utf-8") as writer:
        json.dump(init_mt, writer)

    # save the transition matrix
    with open(tran_sv, "w", encoding="utf-8") as writer:
        json.dump(tran_mt, writer)

    # save the emission matrix
    with open(emis_sv, "w", encoding="utf-8") as writer:
        json.dump(emis_mt, writer)


def matrix_load() -> (dict, dict, dict):
    """
    load the 3 matrix from the specified file
    """
    # load the initial matrix
    with open(init_sv, "r", encoding="utf-8") as reader:
        init_mt = json.load(reader)

    # load the transition matrix
    with open(tran_sv, "r", encoding="utf-8") as reader:
        tran_mt = json.load(reader)

    # load the emission matrix
    with open(emis_sv, "r", encoding="utf-8") as reader:
        emis_mt = json.load(reader)

    return init_mt, tran_mt, emis_mt


def matrix_normal(init_mt: dict, tran_mt: dict, emis_mt: dict) -> None:
    """
    normalize a matrix on the row level
    """
    # normalize the initial matrix
    normalize_dict(init_mt)

    # normalize the transition matrix
    for state in tran_mt:
        normalize_dict(tran_mt[state])

    # normalize the emission matrix
    for state in emis_mt:
        normalize_dict(emis_mt[state])
