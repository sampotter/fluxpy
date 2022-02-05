import matplotlib.pyplot as plt
import meshpy.triangle as triangle
import numpy as np

from scipy.constants import sigma
from scipy.optimize import brentq
from scipy.interpolate import interp1d

class HemisphericalCrater:

    def __init__(self, beta, rc, e0, F0, rho, emiss):
        self.beta = beta # angle between crater bottom and crater rim measured from sphere center
        self.rc = rc # radius of crater in the ground plane
        self.e0 = e0 # elevation of sun above the ground plane
        self.F0 = F0 # solar constant
        self.rho = rho # albedo
        self.emiss = emiss # emissivity

        # Parameters of sphere defining the crater
        self.H = rc/np.tan(beta) # height of sphere center above ground plane
        self.r = rc/np.sin(beta) # radius of sphere defining crater
        self.Sc = 2*np.pi*self.r*(self.r - self.H) # surface area of crater

        # Geometric constants
        self.f = (1 - np.cos(self.beta))/2
        self.b = self.f*(self.emiss + self.rho*(1 - self.f))/(1 - self.rho*self.f)

        # Compute the groundtruth temperature in the shadowed region
        self.T_gt = F0*np.sin(e0)*self.f
        self.T_gt *= (1 - rho)/(1 - rho*self.f)
        self.T_gt *= 1 + rho*(1 - self.f)/emiss
        self.T_gt /= sigma
        self.T_gt **= 0.25

    @property
    def t0(self):
        return np.tan(self.e0)

    def _x_silhouette(self, y):
        '''A function that computes the x coordinate of a point on the shadow
line for a given y coordinate. There's probably a smarter and more
robust way to do this...

        '''
        if abs(y) > self.rc:
            return None
        x0 = np.sqrt(self.rc**2 - y**2)
        def f(x):
            z = self.H - np.sqrt(self.r**2 - x**2 - y**2)
            dx = x0 - x
            return z + self.t0*dx
        X = np.linspace(-x0, x0, 21)
        F = f(X)
        I = np.argsort(abs(F))[:2]
        i = np.argsort(abs(X[I] - x0))[1]
        xm = X[I[i]]
        dxm = min(abs(xm - x0), abs(xm + x0))/2
        if np.sign(f(xm - dxm)) == np.sign(f(xm + dxm)):
            return None
        return brentq(f, xm - dxm, xm + dxm)

    def make_trimesh(self,h,contour_rim=False,contour_shadow=False,**kwargs):
        '''TODO: document me!'''
        if contour_rim and contour_shadow:
            return self._make_trimesh__contour_rim_and_shadow(h, **kwargs)
        if contour_rim:
            return self._make_trimesh__contour_rim(h, **kwargs)
        if contour_shadow:
            raise ValueError("can only contour the shadow line if the crater rim is also contoured")
        else:
            return self._make_trimesh__no_contouring(h, **kwargs)

    def _get_z(self, x, y):
        Rc = np.sqrt(x**2 + y**2)
        if Rc <= self.rc + np.finfo(np.float32).eps:
            return self.H - np.sqrt(self.r**2 - Rc**2)
        else:
            return 0

    def _get_refinement_func(self, h):
        max_area = (2/3)*h**2
        def refinement_func(verts, _):
            verts = np.array(verts)
            assert verts.shape[0] == 3
            P = np.array([(x, y, self._get_z(x, y)) for x, y in verts])
            area = np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0]))/2
            return area > max_area
        return refinement_func

    def _make_trimesh__no_contouring(self, h, save_plots=False):
        if save_plots:
            raise RuntimeError('not implemented yet')

        points = np.array(
            [(1, 1), (-1, 1), (-1, -1), (1, -1)], dtype=np.float32)
        facets = np.array([(0, 1), (1, 2), (2, 3), (3, 0)], np.intc)

        info = triangle.MeshInfo()
        info.set_points(points)
        info.set_facets(facets)

        should_refine = self._get_refinement_func(h)
        mesh = triangle.build(info, refinement_func=should_refine)

        xy = np.array(mesh.points)
        F = np.array(mesh.elements)
        V = np.array([(x, y, self._get_z(x, y)) for x, y in zip(*xy.T)])

        return V, F

    def _make_trimesh__contour_rim(self, h, return_part_indices=False,
                                   return_parts=False, save_plots=False):
        if save_plots:
            raise RuntimeError('not implemented yet')

        ncirc = int(np.ceil(2*np.pi*self.rc/h))
        theta = np.linspace(0, 2*np.pi, ncirc, endpoint=False)
        xcirc = self.rc*np.cos(theta)
        ycirc = self.rc*np.sin(theta)

        points_rim = np.array([xcirc, ycirc]).T
        facets_rim = np.array([np.arange(ncirc), np.mod(np.arange(ncirc) + 1, ncirc)]).T

        should_refine = self._get_refinement_func(h)

        points_bd = np.array(
            [(1, 1), (-1, 1), (-1, -1), (1, -1)], dtype=np.float32)
        facets_bd = np.array([(0, 1), (1, 2), (2, 3), (3, 0)], np.intc)

        points_plane = np.concatenate([points_bd, points_rim], axis=0)
        facets_plane = np.concatenate([facets_bd, facets_rim + facets_bd.shape[0]], axis=0)

        info_plane = triangle.MeshInfo()
        info_plane.set_points(points_plane)
        info_plane.set_facets(facets_plane)
        info_plane.set_holes([(0, 0)])

        mesh_plane = triangle.build(info_plane, refinement_func=should_refine)
        V_plane = np.array([(x, y, self._get_z(x, y)) for x, y in mesh_plane.points])
        F_plane = np.array(mesh_plane.elements)

        num_verts_plane = V_plane.shape[0]
        num_faces_plane = F_plane.shape[0]

        info_crater = triangle.MeshInfo()
        info_crater.set_points(points_rim)
        info_crater.set_facets(facets_rim)

        mesh_crater = triangle.build(info_crater, refinement_func=should_refine)
        V_crater = np.array([(x, y, self._get_z(x, y)) for x, y in mesh_crater.points])
        F_crater = np.array(mesh_crater.elements)

        num_faces_crater = F_crater.shape[0]

        V = np.concatenate([V_plane, V_crater], axis=0)
        F = np.concatenate([F_plane, F_crater + num_verts_plane], axis=0)

        assert F.max() == V.shape[0] - 1

        num_faces = F.shape[0]
        assert num_faces == num_faces_plane + num_faces_crater

        inds = (0, num_faces_plane, num_faces)

        if return_part_indices:
            return V, F, inds
        elif return_parts:
            parts = (
                np.arange(inds[0], inds[1]),
                np.arange(inds[1], inds[2])
            )
            return V, F, parts
        else:
            return V, F

    def _make_trimesh__contour_rim_and_shadow(self, h,
                                              return_part_indices=False,
                                              return_parts=False,
                                              save_plots=False):
        if return_part_indices and return_parts:
            raise Exception('''
only one of return_part_indices or return_parts should be true''')

        # We want our triangle mesh to conform to the shadow
        # line. This takes a little doing.

        # First, find the location of the points where the shadow line
        # and the rim of the crater intersects, (xp, yp) and (xp,
        # -yp).
        yp = np.sqrt((self.t0**2 + 1)*self.rc**2 - self.t0**2*self.r**2)
        xp = np.sqrt(self.rc**2 - yp**2)

        # Next, discretize the shadow line using by uniformly sampling
        # the y coordinate. This will be nonuniform with respect to
        # the arc length of the shadow line, which we will fix.
        Yp = np.linspace(-yp, yp, int(np.ceil(2*yp/h)))
        Xp = np.empty_like(Yp)
        Xp[0] = xp
        Xp[-1] = xp
        Xp[1:-1] = np.array([self._x_silhouette(y) for y in Yp[1:-1]])

        # In this iteration, we continually reparametrize the
        # piecewise linear curve connecting the points (Xp, Yp) until
        # the length of each individual segment is the same up to a
        # small tolerance
        while True:
            # NOTE: The method I'm using right now for finding the x
            # coordinates of the silhouette doesn't work very well
            # near the crater rim, and will set values of Xp to NaN
            # when it fails. If this happens, just compute these
            # values by interpolation.
            isnan = np.isnan(Xp)
            if isnan.any():
                Xp[isnan] = interp1d(
                    Yp[~isnan], Xp[~isnan], kind='cubic')(Yp[isnan])

            # Integrate along the parameter of the piecewise linear
            # curve discretizing the shadow line and find the number
            # of points that we would need to use to discretize it
            # with roughly O(h) fineness.
            dL = np.sqrt((Xp[1:] - Xp[:-1])**2 + (Yp[1:] - Yp[:-1])**2)
            dL = np.concatenate([[0], dL])
            L = np.cumsum(dL)
            n = int(np.ceil(L[-1]/h))

            # Check for convergence
            param_error = dL[1:].max() - dL[1:].mean()
            if param_error < np.finfo(np.float32).resolution:
                break # We've converged!

            # Attempt to reparametrize the curve uniformly. The
            # resulting segments may not have exactly the same length,
            # but it will be more uniform than before.
            Yl = np.empty((n,), dtype=np.float64)
            Xl = np.empty_like(Yl)
            j = 1
            for i, l in enumerate(np.linspace(0, L[-1], n)):
                while l < L[j - 1] or L[j] < l:
                    j += 1
                s = (l - L[j - 1])/dL[j]
                Yl[i] = (1 - s)*Yp[j - 1] + s*Yp[j]
                Xl[i] = self._x_silhouette(Yl[i])
            Xl[0] = xp
            Xl[-1] = xp

            # Prepare for the next iteration
            Xp = Xl.copy()
            Yp = Yl.copy()

        # Next, we need to discretize the two parts of the crater. We do this
        # by finding the angle to (xp, yp), and sampling the arc on either
        # side of (xp, +/- yp).

        # The angle where (xp, yp) lies on the crater rim
        phi = np.arctan(yp/xp)

        # Discretize the right-hand side of the rim
        theta = np.linspace(-phi, phi, int(np.ceil(2*self.rc*phi/h)))
        Xcirc1 = self.rc*np.cos(theta)
        Ycirc1 = self.rc*np.sin(theta)

        # Discretize the left-hand side of the rim
        theta = np.linspace(
            phi, 2*np.pi - phi, int(np.ceil(2*self.rc*(np.pi - phi)/h)))
        Xcirc2 = self.rc*np.cos(theta)
        Ycirc2 = self.rc*np.sin(theta)

        # Define our refinement function which will be passed to
        # Shewchuk's triangle. If we wanted to do something more
        # sophisticated, we could try some other definitions here.
        should_refine = self._get_refinement_func(h)

        if save_plots:
            fig = plt.figure(figsize=(8.4, 4.9))

        # First, mesh the upper horizontal plane. Start by adding a
        # square boundary.
        points = np.array(
            [(1, 1), (-1, 1), (-1, -1), (1, -1)], dtype=np.float32)
        facets = np.array([(0, 1), (1, 2), (2, 3), (3, 0)], np.intc)

        if save_plots:
            ax = fig.add_subplot(2, 3, 1)
            ax.plot(
                *np.concatenate([points, points[0, :].reshape(1, 2)], axis=0).T,
                linewidth=1, c='k', zorder=1)
            ax.scatter(
                *points.T, s=15, edgecolors='k', facecolors='r', zorder=2)
            ax.plot(
                np.concatenate([Xcirc1, Xcirc2[1:]]),
                np.concatenate([Ycirc1, Ycirc2[1:]]),
                linewidth=1, c='k', zorder=1)
            ax.scatter(
                np.concatenate([Xcirc1, Xcirc2[1:-1]]),
                np.concatenate([Ycirc1, Ycirc2[1:-1]]),
                s=15, edgecolors='k', facecolors='r', zorder=2)
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect('equal')

        points = np.concatenate([
            points,
            np.array([Xcirc1, Ycirc1]).T,
            np.array([Xcirc2[1:-1], Ycirc2[1:-1]]).T,
        ], axis=0)
        ncirc = Xcirc1.size + Xcirc2.size - 2
        facets = np.concatenate([
            facets,
            facets.shape[0] + np.array([
                np.arange(ncirc),
                np.mod(np.arange(ncirc) + 1, ncirc)
            ]).T
        ], axis=0)

        info = triangle.MeshInfo()
        info.set_points(points)
        info.set_holes([(0, 0)])
        info.set_facets(facets)

        mesh = triangle.build(info, refinement_func=should_refine)
        xy = np.array(mesh.points)
        F = np.array(mesh.elements)

        num_faces_ground_plane = len(mesh.elements)

        if save_plots:
            ax = fig.add_subplot(2, 3, 4)
            ax.triplot(*xy.T, triangles=F, linewidth=1, c='k', zorder=1)
            ax.scatter(*xy.T, s=3, c='k', zorder=2)
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect('equal')

        # Next, mesh the shadowed part of the crater
        points = np.concatenate([
            np.array([Xp[::-1], Yp[::-1]]).T,
            np.array([Xcirc1[1:-1], Ycirc1[1:-1]]).T
        ], axis=0)
        npts = Xp.size + Xcirc1.size - 2
        facets = np.array([
            np.arange(npts),
            np.mod(np.arange(npts) + 1, npts)
        ]).T

        if save_plots:
            ax = fig.add_subplot(2, 3, 2)
            ax.plot(
                *np.concatenate([points, points[0, :].reshape(1, 2)], axis=0).T,
                linewidth=1, c='k', zorder=1)
            ax.scatter(
                *points.T, s=15, edgecolors='k', facecolors='r', zorder=2)
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect('equal')

        info = triangle.MeshInfo()
        info.set_points(points)
        info.set_facets(facets)

        mesh = triangle.build(info, refinement_func=should_refine)
        F = np.concatenate([F, xy.shape[0] + np.array(mesh.elements)], axis=0)
        xy = np.concatenate([xy, np.array(mesh.points)], axis=0)

        num_faces_shadowed = len(mesh.elements)

        if save_plots:
            ax = fig.add_subplot(2, 3, 5)
            ax.triplot(
                *np.array(mesh.points).T, triangles=np.array(mesh.elements),
                linewidth=1, c='k', zorder=1)
            ax.scatter(*np.array(mesh.points).T, s=3, c='k', zorder=2)
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect('equal')

        # Now, mesh the other side
        points = np.concatenate([
            np.array([Xp, Yp]).T,
            np.array([Xcirc2[1:-1], Ycirc2[1:-1]]).T
        ], axis=0)
        npts = Xp.size + Xcirc2.size - 2
        facets = np.array([
            np.arange(npts),
            np.mod(np.arange(npts) + 1, npts)
        ]).T

        if save_plots:
            ax = fig.add_subplot(2, 3, 3)
            ax.plot(
                *np.concatenate([points, points[0, :].reshape(1, 2)], axis=0).T,
                linewidth=1, c='k', zorder=1)
            ax.scatter(
                *points.T, s=15, edgecolors='k', facecolors='r', zorder=2)
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect('equal')

        info = triangle.MeshInfo()
        info.set_points(points)
        info.set_facets(facets)

        mesh = triangle.build(info, refinement_func=should_refine)
        F = np.concatenate([F, xy.shape[0] + np.array(mesh.elements)], axis=0)
        xy = np.concatenate([xy, np.array(mesh.points)], axis=0)

        num_faces_illuminated = len(mesh.elements)

        num_faces = F.shape[0]
        assert num_faces_ground_plane + num_faces_illuminated \
            + num_faces_shadowed == num_faces

        if save_plots:
            ax = fig.add_subplot(2, 3, 6)
            ax.triplot(
                *np.array(mesh.points).T, triangles=np.array(mesh.elements),
                linewidth=1, c='k', zorder=1)
            ax.scatter(*np.array(mesh.points).T, s=3, c='k', zorder=2)
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect('equal')

            fig.savefig('ingersoll-meshing.pdf')
            plt.close(fig)

        # Compute the z component of the crater vertices
        Rc = np.sqrt(np.sum(xy**2, axis=1))
        I = Rc <= self.rc + np.finfo(np.float32).eps
        Z = self.H - np.sqrt(self.r**2 - Rc[I]**2)
        assert Z.max() < np.finfo(np.float32).eps
        Z = np.minimum(0, Z)

        # Assemble the vertex array
        V = np.empty((xy.shape[0], 3), dtype=np.float32)
        V[:, :2] = xy
        V[:, 2] = 0
        V[I, 2] = Z

        inds = (
            0,
            num_faces_ground_plane,
            num_faces_ground_plane + num_faces_shadowed,
            num_faces
        )

        if return_part_indices:
            return V, F, inds
        elif return_parts:
            parts = (
                np.arange(inds[0], inds[1]),
                np.arange(inds[1], inds[2]),
                np.arange(inds[2], inds[3])
            )
            return V, F, parts
        else:
            return V, F
