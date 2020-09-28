def get_quadtree_inds(x, y):
    assert x.size == y.size

    if len(x) == 0:
        return np.array([], dtype=int)

    if len(x) == 1:
        return np.array([0], dtype=int)

    xm = (x.max() + x.min())/2
    ym = (y.max() + y.min())/2

    I00 = np.where((x <= xm) & (y <= ym))[0]
    I10 = np.where((x > xm) & (y <= ym))[0]
    I01 = np.where((x <= xm) & (y > ym))[0]
    I11 = np.where((x > xm) & (y > ym))[0]

    I00 = I00[get_quadtree_inds(x[I00], y[I00])]
    I10 = I10[get_quadtree_inds(x[I10], y[I10])]
    I01 = I01[get_quadtree_inds(x[I01], y[I01])]
    I11 = I11[get_quadtree_inds(x[I11], y[I11])]

    I = np.concatenate([I00, I10, I01, I11])

    return I
