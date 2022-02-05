#!/usr/bin/env python

from pathlib import Path

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.plot import plot_blocks

stats_path = Path('stats')

for FF_path in stats_path.glob('eps_*/ingersoll_p*/FF.bin'):
    FF = CompressedFormFactorMatrix.from_file(FF_path)
    fig, ax = plot_blocks(FF._root)
    fig.savefig(FF_path.parent/'blocks.png')
