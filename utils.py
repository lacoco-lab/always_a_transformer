import json

def get_last_write_index(flipflop):
    """
    Get the index of the last write operation in the flipflop string
    :param flipflop: str
    :return: int
    """

    for i in range(len(flipflop)-1, 0, -1):
        if flipflop[i] == 'w':
            return i


def save_to_json(path, list_of_dicts):
    """
    Save a list of dictionaries to a jsonlines file
    :param path: directory to save the jsonlines file
    :param list_of_dicts: data to save
    :return: void
    """

    with open(path + "/results.jsonl", 'w') as out:
        for ddict in list_of_dicts:
            jout = json.dumps(ddict) + '\n'
            out.write(jout)