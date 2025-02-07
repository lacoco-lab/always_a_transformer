import sys
from generate_majority import validate_majority, analyze_majority

"""
This file explains Majority and gives some examples.

Majority is formulated as following:
Given a binary string 'w' of length 'n', decide whether the number of 1s in 'w' is greater than 
the number of 0s.

The complexity of the majority string depends on the percentage of 1s and 0s in the string. We generate
string in ranges of those percentages, in bins. Namely:

Name s_i to be an ith set of your whole majority set. In each s_i, there are validation bins in ranges of k to l, e.g.
bin 1-10, bin 10-20, etc. (corresponding to the min/max percentage of 1s in the string). Inside each bin,
there are strings of length from 10 to 500 (with step 10), that have the corresponding percentage of 1s.
For each length, there are 100 unique strings.

Example data folder structure:
s1
|___b1-10
    |___majority_10.txt
        |___0001000000
        |___0000000100
        |___0000100000
        |___ ....
    |___majority_20.txt
    |___....
|___b10-20
|___b20-30
|___b30-40
|___b40-50
|___b50-60
|___b60-70
|___b70-80
|___b80-90
|___b90-100

========
String example: 00000000000010000010 - length 20, bin (percentage of 1s): 1-10
Other examples in datasets/flipflop/examples.txt
========

To play around, supply a string by running the script as
```
python majority/playground.py 0001000000
```

The script will perform the validation of the string and gives you back statistics about the string.
"""

majority_str = sys.argv[1]

is_valid, response = validate_majority(majority_str, 20, 10, 20)
length, count_ones, percentage_ones = analyze_majority(majority_str)

print(f"Is the string valid? {is_valid}\n{response}")

print(f'The string {majority_str} is of length {len(majority_str)} with {count_ones} counts of 1s'
      f' which corresponds to {percentage_ones} percent.')