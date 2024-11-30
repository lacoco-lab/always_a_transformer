import re

from pathlib import Path

import jsonlines


def get_last_write_index(flipflop):
    """
    Get the index of the last write operation in the flipflop string
    >> get_last_write_index("w0r0")
    0
    >> get_last_write_index("w0w1r0")
    2
    >> get_last_write_index("w0w1r0w1")
    6
    :param flipflop: str
    :return: int
    """

    reversed_flipflop = flipflop[::-1]
    for idx, char in enumerate(reversed_flipflop):
        if char == 'w':
            return len(flipflop) - idx - 1


def save_to_jsonl(path, filename, list_of_dicts):
    """
    Save a list of dictionaries to a jsonlines file
    :param path: directory to save the jsonlines file
    :param list_of_dicts: data to save
    :return: void
    """
    out_path = Path(path, filename)
    with jsonlines.open(out_path, mode='w') as writer:
        writer.write_all(list_of_dicts)


def parse_flipflop_response(response_text):
    try:
        answer = re.search(r'<answer>(.*?)</answer>', response_text).group(1)
        answer = int(answer)
    except (AttributeError, ValueError):
        # print(f"Response: {response_text}")
        answer = -1
    return answer