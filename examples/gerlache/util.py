import itertools as it
import numpy as np

from flux.shape import get_surface_normals, CgalTrimeshShapeModel

def load_stereographic_shape_model(area_str):
    '''shape model in stereographic projection after looking up relevant
data using the "area string"

    '''
    V_st = np.load(f'gerlache_verts_stereo_{area_str}.npy')
    V_st = np.concatenate([V_st, np.ones((V_st.shape[0], 1))], axis=1)
    F = np.load(f'gerlache_faces_{area_str}.npy')
    N_st = get_surface_normals(V_st, F)
    N_st[N_st[:, 2] > 0] *= -1
    return CgalTrimeshShapeModel(V_st, F, N_st)

def raytrace_values(shape_model_st, field, xgrid, ygrid):
    assert field.ndim == 1 and field.size == shape_model_st.num_faces
    dtype = field.dtype
    d = np.array([0, 0, 1], dtype=np.float64)
    m, n = len(xgrid), len(ygrid)
    grid = np.empty((m, n), dtype=dtype)
    grid[...] = np.nan
    for i, j in it.product(range(m), range(n)):
        x = np.array([xgrid[i], ygrid[j], -1], dtype=dtype)
        hit = shape_model_st.intersect1(x, d)
        if hit is not None:
            grid[i, j] = field[hit[0]]
    return grid
