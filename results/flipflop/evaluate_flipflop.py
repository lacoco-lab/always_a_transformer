import argparse
import json


def get_base_accuracy(responses):
    """
    Calculates the baseline (raw) accuracy of the reported answers
    :param responses: list of dictionaries containing model's answers
    :return: float, correct answers proportion
    """

    correct = 0
    for response in responses:
        try:
            if int(response['answer']) == int(response['last_valid_token']):
                correct += 1
        except Exception as e:
            print(f'Invalid response given {response['answer']}')
            continue

    return correct / len(responses)


def get_digit_accuracy(responses, digit):
    """
    Calculates the accuracy of the reported answers
    :param responses: list of dictionaries containing model's answers'
    :param digit: int, which digit to calculate for
    :return: float, correct answers proportion
    """

    flipflops = [response for response in responses if int(response['last_valid_token']) == digit]

    return get_base_accuracy(flipflops)


def get_relaxed_distance_accuracy(dist_beg, dist_end, responses):
    """
    Calculates the accuracy of the model's response for a specified range of distances.
    :param dist_beg: int, beginning of the distance range
    :param dist_end: int, end of the distance range
    :param responses: list of dictionaries containing model's responses
    :return: float, correct answer proportion
    """

    assert dist_beg >= 2 and dist_beg % 2 == 0 and dist_end % 2 == 0
    assert dist_beg < dist_end

    responses_in_distance = []
    for response in responses:
        assert dist_beg + 2 <= len(response['flipflop']) and dist_end + 2 <= len(response['flipflop'])

        if len(response['flipflop']) - 2 - response['last_write_index'] <= dist_end and \
                len(response['flipflop']) - 1 - response['last_write_index'] >= dist_beg:
            responses_in_distance.append(response)

    if len(responses_in_distance) == 0:
        raise ValueError('No responses in the specified distance to last write')

    return get_base_accuracy(responses_in_distance)


def get_strict_distance_accuracy(distance_to_last_w, responses):
    """
    Calculates the accuracy of the model's response for a certain distance between last write and read.
    :param distance_to_last_w: int, between last read and write
    :param responses: list of dictionaries containing model's responses
    :return: float, correct answer proportion
    """

    assert distance_to_last_w >= 2 and distance_to_last_w % 2 == 0

    responses_in_distance = []

    for response in responses:
        assert distance_to_last_w + 2 <= len(response['flipflop'])

        if len(response['flipflop']) - 2 - response['last_write_index'] == distance_to_last_w:
            responses_in_distance.append(response)

    if len(responses_in_distance) == 0:
        raise ValueError('No responses in the specified distance to last write')

    return get_base_accuracy(responses_in_distance)


def get_per_dist_accuracy(responses):
    """
    Calculates the accuracy of the model's response per distance until the last write.'
    :param responses: list of dictionaries containing model's responses'
    :return: dict, correct answer proportion per last write index
    """

    distances = {}
    results = {}

    for response in responses:
        distance = len(response['flipflop']) - response['last_write_index']
        if distance not in distances.keys():
            distances[distance] = []
        distances[distance].append(response)

    for dist, responses in distances.items():

        results[dist] = get_base_accuracy(responses)

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=str, default="llama3.1_70B/distance-worded/s1/flipflop_20_w0_results.jsonl", help="Dir Path to the results")
    ap.add_argument("--dist", type=int, default=4, help="Distance between last write and read instructions")
    ap.add_argument("--digit", type=int, default=0, help="Digit to calculate the accuracy for")
    ap.add_argument("--dist_beg", type=int, default=2, help="Beginning of the distance range for last write and read")
    ap.add_argument("--dist_end", type=int, default=8, help="End of the distance range for last write and read")
    args = ap.parse_args()

    with open(args.path, 'r') as file:
        data = [json.loads(line) for line in file]

    print(f"Baseline accuracy for FlipFlop task is {get_base_accuracy(data)}")
    #print(f"Accuracy for the distance {args.dist} is {get_strict_distance_accuracy(args.dist, data)}")
    #print(f"Accuracy for the digit {args.digit} is {get_digit_accuracy(data, args.digit)}")
    #print(f"Accuracy for the range in distance {args.dist_beg}-{args.dist_end} is {get_relaxed_distance_accuracy(args.dist_beg, args.dist_end, data)}")