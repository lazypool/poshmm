"""
the module used to save & read the test result
"""


from datetime import datetime
import os

SAVE_PATH = os.path.abspath(".")
RESULTS_PATH = SAVE_PATH + "/results.txt"


def summary_test_result(rgh_count: int, wrg_count: int, err_sample: dict) -> None:
    """
    given the right counts, wrong counts adn err_sample then acculate its accury and recorde it
    """
    # return if no answer
    if rgh_count == 0 and wrg_count == 0:
        return

    # the info of test time
    time_info = f"Last tested on {datetime.now()}.\n"

    # the info of test result
    total_info = f"Total number of test samples is {rgh_count + wrg_count}\n"
    compare_info = f"There are {rgh_count} correctly tagged and {wrg_count} incorrectly.\n"
    acc_info = f"Accuracy is {rgh_count / (rgh_count + wrg_count):.2%}\n"

    # the info of the err samples
    col, err_info = 0, "Error prediction distribution:\n"
    for err, num in err_sample.items():
        err_info += f"{err}:{num}"
        if col < 7:
            err_info += "\t"
            col += 1
        else:
            err_info += "\n"
            col = 0

    # summary the info before
    summary_info = time_info + total_info + compare_info + acc_info + err_info
    print(summary_info)

    # save the summary
    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        file.write(summary_info)


def read_test_result() -> None:
    """
    read the info saved in the test result file and do nothing
    """
    try:
        with open(RESULTS_PATH, "r", encoding="utf-8") as file:
            print("Test results has been searched.")
            print(file.read(), "\n")
    except FileNotFoundError:
        print("No existing test results. Skip reading this.\n")
