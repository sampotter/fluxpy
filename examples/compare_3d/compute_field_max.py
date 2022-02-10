#!/usr/bin/env python

import numpy as np
import sys

p_max = -np.infty

for p_path in sys.argv[1:-1]:
    p = np.load(p_path)
    p_max = max(p_max, p.max())

with open(sys.argv[-1], 'w') as f:
    print('%0.16g' % p_max, file=f)
