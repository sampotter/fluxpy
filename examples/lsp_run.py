import logging
import os
from matplotlib import pyplot as plt

from examples.lsp_form_factor_matrix import setup_form_factor_matrix
from examples.lsp_make_shape_model import make_shape_model
from examples.lsp_spice import illuminate_form_factor
from flux.plot import tripcolor_vector

# run sequence from grd to FF
# TODO replace with Norbert's generate_FF_from_grd ?

# compute shape_model from input grd
path = os.path.join('.', 'LDEM_80S_150M_adjusted.grd')
make_shape_model(grdpath=path)

# compute both full and compressed FF
d = {}
for compressed,v in enumerate(["full","compressed"]):
    # only compute new FF if it doesn't exist (possibly an option would be better?)
    if not os.path.exists(f'lsp_{v}_form_factors.bin'):
        logging.warning(f"Generating a new {f'lsp_{v}_form_factors.bin'}")
        setup_form_factor_matrix(compress=compressed)
    d[v] = illuminate_form_factor(FF_path = f'lsp_{v}_form_factors.bin', compressed=compressed)

# compute residuals
V = d['full']['V']
F = d['full']['F']
residuals = d['full']['T'] - d['compressed']['T']

# plot residuals
for i in range(residuals.shape[1]):
    # print('frame = %d' % i)
    fig, ax = tripcolor_vector(V, F, residuals[:, i])
    fig.savefig(f"./frames/lsp_d1_%03d.png" % i)
    plt.close(fig)
