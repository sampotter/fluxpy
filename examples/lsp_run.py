import logging
import os
from matplotlib import pyplot as plt
import numpy as np
from examples.lsp_form_factor_matrix import setup_form_factor_matrix
from examples.lsp_make_shape_model import make_shape_model
from examples.lsp_illuminate import illuminate_form_factor
from flux.linalg2 import xSVDcomputation
from flux.model import update_incoming_radiances_wsvd
from flux.plot import tripcolor_vector

# run sequence from grd to FF
# TODO replace with Norbert's generate_FF_from_grd ?

recompute_FF = True
test_svd = False

# compute shape_model from input grd
path = os.path.join('.', 'LDEM_80S_150M_adjusted.grd')
make_shape_model(grdpath=path)

# compute both full and compressed FF
d = {}
for compressed,v in enumerate(["full", "compressed"]): #
    print(compressed,v)
    # only compute new FF if it doesn't exist (possibly an option would be better?)
    if not os.path.exists(f'lsp_{v}_form_factors.bin') or recompute_FF:
        logging.warning(f"Generating a new {f'lsp_{v}_form_factors.bin'}")
        setup_form_factor_matrix(compress=compressed, tol=5e-2)
    try:
        d[v] = illuminate_form_factor(FF_path = f'lsp_{v}_form_factors.bin', compressed=compressed)
    except:
        d[v] = illuminate_form_factor(FF_path=f'lsp_{v}_form_factors.bin', compressed=compressed)

if test_svd:
    xSVDcomputation(f'lsp_full_form_factors.bin', TRUNC=200, mode='approx')
    d["svd"] = illuminate_form_factor(FF_path = f'lsp_full_form_factors.bin', compressed=False, use_svd=True)

# compute residuals
V = d['full']['V']
F = d['full']['F']
if test_svd:
    residuals = np.abs(d['full']['Qrefl']-d['svd']['Qrefl']) #  #
else:
    residuals = np.abs(d['full']['Qrefl']-d['compressed']['Qrefl']) #  #

print(len(residuals))
step2_res = residuals[:,8]
print(np.where(step2_res>1.e14))
# exit()

# plot residuals
for i in range(residuals.shape[1]):
    # print('frame = %d' % i)
    fig, ax = tripcolor_vector(V, F, residuals[:, i], vmax=1.e14)
    fig.savefig(f"./frames/lsp_dfc32_%03d.png" % i)
    plt.close(fig)

    fig, ax = tripcolor_vector(V, F, d['full']['Qrefl'][:, i], vmax=1.e16)
    fig.savefig(f"./frames/lsp_fd32_%03d.png" % i)
    plt.close(fig)

    fig, ax = tripcolor_vector(V, F, d['compressed']['Qrefl'][:, i], vmax=1.e16)
    fig.savefig(f"./frames/lsp_cd32_%03d.png" % i)
    plt.close(fig)

    if test_svd:
        fig, ax = tripcolor_vector(V, F, d['svd']['Qrefl'][:, i], vmax=1.e16)
        fig.savefig(f"./frames/lsp_sd32_%03d.png" % i)
        plt.close(fig)