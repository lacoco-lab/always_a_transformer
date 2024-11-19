import sys
from flipflop_generator import validate_flip_flop

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

flipflop_str = sys.argv[1]

validate_flip_flop(flipflop_str)