import logging
import os
from matplotlib import pyplot as plt
import numpy as np
import pyvista as pv

from examples.lsp_form_factor_matrix import setup_form_factor_matrix
from examples.lsp_make_shape_model import make_shape_model
from flux.compressed_form_factors import CompressedFormFactorMatrix
import flux.form_factors as ff
from flux.plot import tripcolor_vector, plot_blocks
from flux.util import tic

recompute_FF = True

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
        setup_form_factor_matrix(compress=compressed, tol=0., min_size=1.e9)

cFF = CompressedFormFactorMatrix.from_file('lsp_compressed_form_factors.bin')
fFF = ff.FormFactorMatrix.from_file('lsp_full_form_factors.bin')#.tocoo()

print("cFF shape/depth:", cFF.shape, cFF.depth)
print("fFF shape:", fFF.shape)

tic()
fig, ax = plot_blocks(cFF._root)
fig.savefig('blocks.png')
plt.close(fig)
print('- wrote blocks.png')

# TODO replace this horror by an efficient residuals computation
# (is this correct AT ALL, btw? Not sure...)
residuals = []
e = np.zeros(cFF.shape[1])
for i in range(cFF.shape[1]):
    e[i] = 1
    tmp = cFF@e - fFF.getrow(i).toarray()
    residuals.append(tmp)
    e *= 0.

residuals = np.vstack(residuals)
print("norm of cFF - fFF = ", np.linalg.norm(residuals))

# plot residuals
tri_mesh = pv.make_tri_mesh(cFF.shape_model.V, cFF.shape_model.F)
tri_mesh.cell_arrays['form factor'] = residuals

plotter = pv.Plotter()
plotter.add_mesh(tri_mesh, lighting=False) #, cmap=cc.cm.coolwarm)
plotter.show()
