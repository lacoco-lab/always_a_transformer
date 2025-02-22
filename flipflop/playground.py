import sys
import os
import glob
from flipflop_generator import validate_flip_flop, validate_replaced_flip_flop, validate_before_first

"""
This file explains FlipFlop and gives some examples.

FlipFlop is a family of formal languages introduced in the paper https://arxiv.org/abs/2306.00946
as a proxy for closed-domain hallucinations.

========
String example: w0w1i1i0r1
Other examples in datasets/flipflop/examples.txt
========

A canonical form of the language abides the following rules:
(i) The first instruction is always write (w), and the last instruction is always read.
(ii) The other instructions are drawn according to given probabilities (pw,pr,pi) with pi = 1−pw−pr.
(iii) The nondeterministic data symbols (paired with w or i, 1 or 0 in our case) 
are drawn i.i.d. and uniformly.

To play around, supply a string by running the script as
```
python flipflop/playground.py w0w1i1i0r0
```

The script will perform the validation of the string and raise an error in case the string is not valid 
according to the canonical form.
"""

path = '../datasets/flipflop/before-first/s5'

for filename in glob.glob(os.path.join(path, '*.txt')):
    data = []
    valid_count = 0
    with open(os.path.join(os.getcwd(), filename), 'r') as f:
        for line in f:
            data.append(line.strip())
            
        for line in data:
            validate_before_first(line)
            valid_count += 1
        
        print(f'Finished validating file {filename}. {valid_count} strings valid out of {len(data)}')