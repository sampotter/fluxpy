import numpy as np
import trimesh

from flux.shape import CgalTrimeshShapeModel

def shape_model_from_obj_file(path):
    mesh = trimesh.load(path)
    verts = np.array(mesh.vertices).astype(np.float32)
    faces = np.array(mesh.faces).astype(np.uintp)
    normals = np.array(mesh.face_normals).astype(np.float32)
    return CgalTrimeshShapeModel(verts, faces, normals)
