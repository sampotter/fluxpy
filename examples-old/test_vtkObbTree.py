import meshzoo
import vtk
import vtk.util.numpy_support

V, F = meshzoo.icosa_sphere(5)

verts = vtk.vtkPoints()
verts.SetData(vtk.util.numpy_support.numpy_to_vtk(V))

faces = vtk.vtkCellArray()
faces.SetCells(F.shape[0], vtk.util.numpy_support.numpy_to_vtkIdTypeArray(F))

poly_data = vtk.vtkPolyData()
poly_data.SetPoints(verts)
poly_data.SetPolys(faces)

obb_tree = vtk.vtkOBBTree()
# obb_tree.SetDataSet(poly_data)
# obb_tree.BuildLocator()
