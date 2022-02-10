#!/usr/bin/env python

import sys

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.plot import plot_blocks

FF = CompressedFormFactorMatrix.from_file(sys.argv[1])

fig, ax = plot_blocks(FF._root)

fig.savefig(sys.argv[2])
fig.savefig(sys.argv[3])
