#!/usr/bin/env python

import matplotlib.pyplot as plt

from glob import glob
from pathlib import Path

from flux.compressed_form_factors import CompressedFormFactorMatrix
from flux.plot import plot_blocks

for FF_path in glob('FF_*_*_*.bin'):
    print(f'- {FF_path}')
    FF = CompressedFormFactorMatrix.from_file(FF_path)
    fig, ax = plot_blocks(FF._root)
    plot_path = FF_path[:-4] + '.png'
    fig.savefig(Path('gerlache_plots')/plot_path)
    plt.close(fig)
