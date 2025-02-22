import random


def generate_flip_flop_with_distance(length, w_idx):
    """
    Generate valid FlipFlop string by specifying the distance between last write and read
    :param length: int, length of the generated string
    :param w_idx: int, at which index you want your write to be
    :return: str, valid flipflop string
    """

    # sanity checks for length
    assert length % 2 == 0  and length >= 4 and length-4 >= 0
    assert w_idx % 2 == 0 and w_idx <= length-4

    flipflop_str = ['w', random.choice(['0', '1'])]
    last_written_bit = flipflop_str[1]

    for i in range(2, length - 2, 2):

        if i == w_idx:
            instruction = 'w'
            data_bit = random.choice(['0', '1'])
            last_written_bit = data_bit
        else:
            instruction = random.choice(['r', 'i'])

            if instruction == 'r':
                data_bit = last_written_bit
            elif instruction == 'i':
                data_bit = random.choice(['0', '1'])

        flipflop_str.append(instruction)
        flipflop_str.append(data_bit)

    flipflop_str.append('r')
    flipflop_str.append(last_written_bit)

    return ''.join(flipflop_str)


def generate_replaced_flip_flop(length, prob_w, prob_r, digit_0='a', digit_1='b', write='c', read='d', ignore='e'):
    """
    Generate valid FlipFlop string:
    1. always begins with 'w', ends with 'r'
    2. T >= 4

    :param length: int, length of generated string
    :param prob_w: float, probability of write instruction
    :param prob_r: float, probability of read instruction
    :return: str, valid flipflop string
    """

    assert length % 2 == 0  # the flipflop string must be even, because all instructions come with a data bit
    assert prob_w + prob_r <= 1  # sanity check for the probabilities
    assert length >= 4  # sanity check for the length

    prob_i = 1 - prob_w - prob_r

    flipflop_str = [write, random.choice([digit_0, digit_1])]
    last_written_bit = flipflop_str[1]

    for i in range(2, length-2, 2):
        instruction = random.choices([write, read, ignore], weights=[prob_w, prob_r, prob_i], k=1)[0]

        if instruction == read:
            data_bit = last_written_bit
        elif instruction == ignore:
            data_bit = random.choice([digit_0, digit_1])
        elif instruction == write:
            data_bit = random.choice([digit_0, digit_1])
            last_written_bit = data_bit

        flipflop_str.append(instruction)
        flipflop_str.append(data_bit)

    flipflop_str.append(read)
    flipflop_str.append(last_written_bit)

    return ''.join(flipflop_str)


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

    assert length % 2 == 0  # the flipflop string must be even, because all instructions come with a data bit
    assert prob_w + prob_r <= 1  # sanity check for the probabilities
    assert length >= 4  # sanity check for the length

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


def generate_all_valid_flipflops(length):
    """
    Generate all possible valid FlipFlop strings of a given length.
    :param length: int, length of the FlipFlop string (must be even)
    :return: list of str, all valid FlipFlop strings of the given length
    """

    if length < 4 or length % 2 != 0:
        raise ValueError("Length must be an even number and at least 4.")

    def backtrack(current, last_written_bit):
        # Base case: if the string reaches the desired length
        if len(current) == length - 2:
            return [current + ['r', last_written_bit]]

        results = []
        for instruction in ['w', 'r', 'i']:
            if instruction == 'r':
                results.extend(backtrack(current + [instruction, last_written_bit], last_written_bit))
            elif instruction == 'w':
                for bit in ['0', '1']:
                    results.extend(backtrack(current + [instruction, bit], bit))
            elif instruction == 'i':
                for bit in ['0', '1']:
                    results.extend(backtrack(current + [instruction, bit], last_written_bit))

        return results

    valid_flipflops = []
    for first_bit in ['0', '1']:
        valid_flipflops.extend(backtrack(['w', first_bit], first_bit))

    return [''.join(flipflop) for flipflop in valid_flipflops]


def generate_relaxed_flip_flop(length, prob_w, prob_r):
    """
    Generate valid FlipFlop string:
    1. always begins with 'r' or 'i', and ends with 'r'
    2. T >= 4

    :param length: int, length of generated string
    :param prob_w: float, probability of write instruction
    :param prob_r: float, probability of read instruction
    :return: str, valid flipflop string
    """

    assert length % 2 == 0  # the flipflop string must be even, because all instructions come with a data bit
    assert prob_w + prob_r <= 1  # sanity check for the probabilities
    assert length >= 4  # sanity check for the length

    prob_i = 1 - prob_w - prob_r

    # Ensuring equal probability of starting prefixes, excluding 'w'
    first_instruction = random.choice(['r', 'i'])
    first_data_bit = random.choice(['0', '1'])
    flipflop_str = [first_instruction, first_data_bit]
    last_written_bit = first_data_bit

    for i in range(2, length - 2, 2):
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


def generate_relaxed_flip_flop(length, prob_w, prob_r, digit_0='a', digit_1='b', write='c', read='d', ignore='e'):
    """
    Generate valid FlipFlop string:
    1. always begins with 'r' or 'i', and ends with 'r'
    2. T >= 4

    :param length: int, length of generated string
    :param prob_w: float, probability of write instruction
    :param prob_r: float, probability of read instruction
    :return: str, valid flipflop string
    """

    assert length % 2 == 0  # the flipflop string must be even, because all instructions come with a data bit
    assert prob_w + prob_r <= 1  # sanity check for the probabilities
    assert length >= 4  # sanity check for the length

    prob_i = 1 - prob_w - prob_r

    # Ensuring equal probability of starting prefixes, excluding 'w'
    first_instruction = random.choice([read, ignore])
    first_data_bit = random.choice([digit_0, digit_1])
    flipflop_str = [first_instruction, first_data_bit]
    last_written_bit = first_data_bit

    for i in range(2, length - 2, 2):
        instruction = random.choices([write, read, ignore], weights=[prob_w, prob_r, prob_i], k=1)[0]

        if instruction == read:
            data_bit = last_written_bit
        elif instruction == ignore:
            data_bit = random.choice([digit_0, digit_1])
        elif instruction == write:
            data_bit = random.choice([digit_0, digit_1])
            last_written_bit = data_bit

        flipflop_str.append(instruction)
        flipflop_str.append(data_bit)

    flipflop_str.append(read)
    flipflop_str.append(last_written_bit)
    return ''.join(flipflop_str)


def generate_all_valid_relaxed_flipflops(length):
    """
    Generate all possible valid FlipFlop strings of a given length.
    :param length: int, length of the FlipFlop string (must be even)
    :return: list of str, all valid FlipFlop strings of the given length
    """

    if length < 4 or length % 2 != 0:
        raise ValueError("Length must be an even number and at least 4.")

    def backtrack(current, last_written_bit):
        # Base case: if the string reaches the desired length
        if len(current) == length - 2:
            return [current + ['r', last_written_bit]]

        results = []
        for instruction in ['w', 'r', 'i']:
            if instruction == 'r':
                results.extend(backtrack(current + [instruction, last_written_bit], last_written_bit))
            elif instruction == 'w':
                for bit in ['0', '1']:
                    results.extend(backtrack(current + [instruction, bit], bit))
            elif instruction == 'i':
                for bit in ['0', '1']:
                    results.extend(backtrack(current + [instruction, bit], last_written_bit))

        return results

    valid_flipflops = []
    for first_instruction in ['r', 'i']:
        for first_bit in ['0', '1']:
            valid_flipflops.extend(backtrack([first_instruction, first_bit], first_bit))

    return [''.join(flipflop) for flipflop in valid_flipflops]


def validate_replaced_flip_flop(flipflop_str, digit_1='a', digit_0 ='b', write='w', read='r', ignore='i'):
    """
    Check if the FlipFlop string is valid
    :param flipflop_str: str, FlipFlop string to check
    :return: True or raise Validation error
    """
    assert len(flipflop_str) >= 4 and len(flipflop_str) % 2 == 0
    assert (flipflop_str[0] == write or flipflop_str[0] == read or flipflop_str[0] == ignore) and flipflop_str[len(flipflop_str)-2] == read
    assert set(flipflop_str).issubset({digit_1, digit_0, write, read, ignore})

    last_displayed_bit = flipflop_str[len(flipflop_str)-1]
    assert last_displayed_bit in [digit_0, digit_1]

    reversed_flipflop = flipflop_str.strip()[::-1]
    current_idx = 1
    displayed_bit = last_displayed_bit

    while current_idx < len(reversed_flipflop):
        if reversed_flipflop[current_idx] == read:
            displayed_bit = reversed_flipflop[current_idx-1]
        elif reversed_flipflop[current_idx] == write:
            assert displayed_bit == reversed_flipflop[current_idx-1], (f"Read bit {displayed_bit} does not match the "
                                                                       f"written bit "
                                                                       f"{reversed_flipflop[current_idx-1]} "
                                                                       f"at index {len(flipflop_str)-current_idx+1}.")
        current_idx += 2

    return True

def validate_before_first(flipflop_str, digit_1='1', digit_0 ='0', write='w', read='r', ignore='i'):
    """
    Check if the FlipFlop string is valid
    :param flipflop_str: str, FlipFlop string to check
    :return: True or raise Validation error
    """
    assert len(flipflop_str) >= 4 and len(flipflop_str) % 2 == 0
    assert (flipflop_str[0] == read or flipflop_str[0] == ignore) and flipflop_str[len(flipflop_str)-2] == read
    assert set(flipflop_str).issubset({digit_1, digit_0, write, read, ignore})

    last_displayed_bit = flipflop_str[len(flipflop_str)-1]
    assert last_displayed_bit in [digit_0, digit_1]

    reversed_flipflop = flipflop_str.strip()[::-1]
    current_idx = 1
    displayed_bit = last_displayed_bit

    while current_idx < len(reversed_flipflop):
        if reversed_flipflop[current_idx] == read:
            displayed_bit = reversed_flipflop[current_idx-1]
        elif reversed_flipflop[current_idx] == write:
            assert displayed_bit == reversed_flipflop[current_idx-1], (f"Read bit {displayed_bit} does not match the "
                                                                       f"written bit "
                                                                       f"{reversed_flipflop[current_idx-1]} "
                                                                       f"at index {len(flipflop_str)-current_idx+1}.")
        current_idx += 2

    return True


def validate_flip_flop(flipflop_str):
    """
    Check if the FlipFlop string is valid
    :param flipflop_str: str, FlipFlop string to check
    :return: True or raise Validation error
    """

    assert len(flipflop_str) >= 4 and len(flipflop_str) % 2 == 0
    assert (flipflop_str[0] == 'w' or flipflop_str[0] == 'r' or flipflop_str[0] == 'i') and flipflop_str[len(flipflop_str)-2] == 'r'
    assert set(flipflop_str).issubset({'w', 'r', 'i', '1', '0'})

    last_displayed_bit = flipflop_str[len(flipflop_str)-1]
    assert last_displayed_bit in ['0', '1']

    reversed_flipflop = flipflop_str.strip()[::-1]
    current_idx = 1
    displayed_bit = last_displayed_bit

    while current_idx < len(reversed_flipflop):
        if reversed_flipflop[current_idx] == 'r':
            displayed_bit = reversed_flipflop[current_idx-1]
        elif reversed_flipflop[current_idx] == 'w':
            assert displayed_bit == reversed_flipflop[current_idx-1], (f"Read bit {displayed_bit} does not match the "
                                                                       f"written bit "
                                                                       f"{reversed_flipflop[current_idx-1]} "
                                                                       f"at index {len(flipflop_str)-current_idx+1}.")
        current_idx += 2

    return True
