"""
this is main.py
"""


import os
import sys
import jieba
from model.learn import train_and_test
from model.predict import predict
from common.records import read_test_result
from common.dataset import SET_PATH, TRAIN_SET, TEST_SET


jieba.setLogLevel(20)


if __name__ == "__main__":
    # give the last train and test info
    read_test_result()

    # check if the matrix is existing
    CHECK1 = os.path.exists("matrix/initial.json")
    CHECK2 = os.path.exists("matrix/transition.json")
    CHECK3 = os.path.exists("matrix/emission.json")
    IS_TRAINED = CHECK1 and CHECK2 and CHECK3
    if IS_TRAINED:
        print("The model has been trained already.")
    else:
        print("The model has not been trained yet.")

    # decide if train a new model
    NEED_TRAIN = input("Want to train a new model? [N/y] ") in ("Y", "y", "yes")

    if NEED_TRAIN:
        print("Training a new model...\n")
        train_and_test(SET_PATH, TRAIN_SET, TEST_SET)
    elif not IS_TRAINED:
        print("Cannot do works below without a trained model. Existing...")
        sys.exit(0)
    else:
        print("Using an old trained model...\n")

    while True:
        sent = input("Please enter a Chinese sentence [quit with 'Q']\n")
        print("\n")

        # Quit the program if enter 'Q'
        if sent == 'Q':
            break

        # Print the result of tagging
        print("INPUTS: ", sent)
        print("OUTPUT: ", predict(sent, jieba.lcut), "\n")
