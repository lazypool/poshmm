"""
some tools for the program, such as normalize, find the max/min
"""

def normalize_dict(freq_dict: dict) -> None:
    """
    given a dict, try to normalize it, do nothing if total value is 0
    """
    total = sum(freq_dict.values())
    if total != 0:
        factor = 1.0 / total
        for key in freq_dict:
            freq_dict[key] *= factor


def cut_phrase(word_state: str) -> (str, str):
    """
    get the word and state from the "word/state" phrase, return ("","") if "/" not found
    """
    seq = word_state.find("/")
    if seq != -1:
        word, state = word_state[:seq], word_state[seq + 1 :]
        if state == "mq":
            state = "m"
        return word, state
    return "", ""


def add_dict(a_dict: dict, b_dict: dict) -> dict:
    """
    add 2 dict, if has the same element then add the value else get the union
    """
    result = a_dict.copy()
    for key, value in b_dict.items():
        if key in result:
            try:
                result[key] += value
            except TypeError:
                result[key] = add_dict(result[key], value)
        else:
            result[key] = value
    return result
