'''Make a quick plot showing the error decay when we use a truncated
SVD to approximate off-diagonal blocks of the form factor matrix. We
can't get the exact bound that we want (i.e., ||yhat - y||/||y|| <=
sigma[k]/sigma[0]), since these off-diagonal blocks are
singular. Instead, we would just like to see that as we blindly choose
the truncation number using this bound, where tol replaces
sigma[k]/sigma[0], we get error convergence like C*tol^p, where p is
approximate 1. It appears that this holds on average, but not for all
test vectors.

'''

import matplotlib.pyplot as plt
import meshzoo
import numpy as np

import flux.compressed_form_factors as cff

from flux.shape import TrimeshShapeModel

tol = 1e-2

V, F = meshzoo.icosa_sphere(15)

# stretch out the sphere
V[:, 0] *= 3
V[:, 1] *= 2
V[:, 2] *= 1

shape_model = TrimeshShapeModel(V, F)

# make surface normals inward facing
outward = (shape_model.P*shape_model.N).sum(1) > 0
shape_model.N[outward] *= -1

FF = cff.CompressedFormFactorMatrix(shape_model, tol=1e-2,
                                    max_rank=30, min_size=400, max_depth=1,
                                    RootBlock=cff.FormFactorOctreeBlock)


# Next, let's check that the error bound actually works...

i, j = 2, 0

Tol = np.logspace(-1, -4, 10)

ErrorMax, ErrorMean = [], []

for tol in Tol:
    print(f'tol: {tol}')

    FFij = FF._root._blocks[i, j].toarray()
    U, S, Vt = np.linalg.svd(FFij, full_matrices=False)
    k = np.where(S/S[0] < tol)[0][0]
    p = 0

    FFij_hat = (U[:, :(k + p)]*S[:(k + p)])@Vt[:(k + p), :]

    num_samp = 100

    X = np.random.randn(FFij_hat.shape[1], num_samp)
    Y = FFij@X
    Y_hat = FFij_hat@X

    Error = np.sqrt(np.sum((Y - Y_hat)**2,
                           axis=1))/np.sqrt(np.sum(Y**2, axis=1))

    print(f'- max error: {abs(Error).max()}')
    print(f'- avg. max error: {abs(Error).mean()}')

    ErrorMax.append(abs(Error).max())
    ErrorMean.append(np.mean(abs(Error)))

ErrorMax, ErrorMean = np.array(ErrorMax), np.array(ErrorMean)

pMax, CMax = np.polyfit(np.log(Tol), np.log(ErrorMax), 1)
print(f'C*tol^p for max |error|: p = {pMax}, C = {CMax}')

pMean, CMean = np.polyfit(np.log(Tol), np.log(ErrorMean), 1)
print(f'C*tol^p for mean |error|: p = {pMean}, C = {CMean}')

plt.figure()
plt.loglog(Tol, ErrorMax, c='k', linestyle='--', linewidth=2, label='max |error|')
plt.loglog(Tol, ErrorMean, c='k', linestyle='-', label='mean |error|')
plt.legend()
plt.tight_layout()
plt.show()
