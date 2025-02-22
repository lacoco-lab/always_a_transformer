import os, glob

path = '../datasets/flipflop/before-first/s5'


def count_before_first(data):
    """
    Count distributions on 0s and 1s in the data
    Suitable for Flip-Flops: before first, after last
    :param data: arr with strings
    :return: dictionary of counts
    """

    counts = {'0': 0, '1': 0}
    for line in data:
        counts[line[line.find('w') - 1]] += 1
        
    return counts

for filename in glob.glob(os.path.join(path, '*.txt')):
    data = []
    with open(os.path.join(os.getcwd(), filename), 'r') as f:
        for line in f:
            data.append(line)
  
        counts = count_before_first(data)
            
        print(f'Filename: {filename} \t Counts: {str(counts)}')