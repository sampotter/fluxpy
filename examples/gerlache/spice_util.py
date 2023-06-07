def get_sunvec(utc0, target, observer, frame, utc1=0, stepet=1, et_linspace=None, path_to_furnsh='aux/simple.furnsh'):
    '''This script uses SPICE to compute a trajectory for the sun, loads a
        shape model discretizing a patch of the lunar south pole (made using
        lsp_make_obj.py), and a compressed form factor matrix for that
        shape model (computed using lsp_compress_form_factor_matrix.py).
        It then proceeds to compute the steady state temperature at each sun
        position, writing a plot of the temperature to disk for each sun
        position.

        '''
    import numpy as np
    import spiceypy as spice

    clktol = '10:000'
    spice.kclear()

    spice.furnsh(path_to_furnsh)
    et0 = spice.str2et(utc0)
    if np.max(et_linspace) == None:
        et1 = spice.str2et(utc1)
        nbet = int(np.ceil((et1 - et0) / stepet))
        et = np.linspace(et0, et1, nbet)
    else:
        # add first epoch to array of time steps
        et = et0 + et_linspace

    # Sun positions over time period
    # TODO I would usually do this, maybe you want to
    # TODO add something similar to the thermal model options/config?
    # possun = spice.spkpos('SUN', et, FluxOpt.get("frame"), 'LT+S', FluxOpt.get("body"))[0]
    possun = spice.spkpos(target, et, frame, 'LT+S', observer)[0]
    possun = possun

    # TODO why am I doing this conversion to lon/lat if then I'm reconstructing the cartesian version?
    lonsun = np.arctan2(possun[:, 1], possun[:, 0])
    lonsun = np.mod(lonsun, 2 * np.pi)
    radsun = np.sqrt(np.sum(possun[:, :] ** 2, axis=1))
    latsun = np.arcsin(possun[:, 2]/radsun)

    sun_vec = radsun[:,np.newaxis] * \
              np.array([
        np.cos(lonsun) * np.cos(latsun),
        np.sin(lonsun) * np.cos(latsun),
        np.sin(latsun)
    ]).T

    return sun_vec
