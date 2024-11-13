import random


def generate_flip_flop(length, prob_w, prob_r):
    """
    Generate valid FlipFlop string:
    1. always begins with 'w', ends with 'r'
    2. T >= 4

    :param length: int, length of generated string
    :param prob_w: float, probability of write instruction
    :param prob_r: float, probability of read instruction
    :return: str, valid flipflop string
    """

    assert length % 2 == 0 # the flipflop string must be even, because all instructions come with a data bit
    assert prob_w + prob_r <= 1 # sanity check for the probabilities
    assert length >= 4 # sanity check for the length

    prob_i = 1 - prob_w - prob_r

    flipflop_str = ['w', random.choice(['0', '1'])]
    last_written_bit = flipflop_str[1]

    for i in range(2, length-2, 2):
        instruction = random.choices(['w', 'r', 'i'], weights=[prob_w, prob_r, prob_i], k=1)[0]

        if instruction == 'r':
            data_bit = last_written_bit
        elif instruction == 'i':
            data_bit = random.choice(['0', '1'])
        elif instruction == 'w':
            data_bit = random.choice(['0', '1'])
            last_written_bit = data_bit

        flipflop_str.append(instruction)
        flipflop_str.append(data_bit)

    flipflop_str.append('r')
    flipflop_str.append(last_written_bit)

    return ''.join(flipflop_str)


def validate_flip_flop(flipflop_str):
    """
    Check if the FlipFlop string is valid
    :param flipflop_str: str, FlipFlop string to check
    :return: None or raise Validation error
    """

    assert len(flipflop_str) >= 4 and len(flipflop_str) % 2 == 0
    assert flipflop_str[0] == 'w' and flipflop_str[len(flipflop_str)-2] == 'r'
    assert set(flipflop_str).issubset({'w', 'r', 'i', '1', '0'})

    displayed_bit = flipflop_str[len(flipflop_str)-1]
    assert displayed_bit in ['0', '1']

    for i in range(len(flipflop_str)-3, -1, -1):
        if flipflop_str[i] == 'w':
            assert displayed_bit == flipflop_str[i+1]
            return None

    raise ValueError(f'FlipFlop string {flipflop_str} is not valid, no "write" instruction was found.')
