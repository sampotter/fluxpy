#include "aabb_wrapper.h"

// This code was originally copied from the example in the CGAL AABB
// documentation on ray shooting:
//
//   https://doc.cgal.org/latest/AABB_tree/index.html

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>

#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>

#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>

#include <iostream>
#include <fstream>

typedef CGAL::Simple_cartesian<double> K;

typedef K::FT FT;
typedef K::Point_3 Point;
typedef K::Vector_3 Vector;
typedef K::Ray_3 Ray;
typedef K::Triangle_3 Triangle;

typedef CGAL::Surface_mesh<Point> Mesh;
typedef boost::graph_traits<Mesh>::face_descriptor face_descriptor;
typedef boost::graph_traits<Mesh>::halfedge_descriptor halfedge_descriptor;

typedef CGAL::AABB_face_graph_triangle_primitive<Mesh> Primitive;
typedef CGAL::AABB_traits<K, Primitive> Traits;
typedef CGAL::AABB_tree<Traits> Tree;

typedef boost::optional<Tree::Intersection_and_primitive_id<Ray>::Type> Ray_intersection;

struct cgal_aabb {
	Mesh mesh;
	Tree tree;
};

void cgal_aabb_alloc(cgal_aabb **aabb) {
	*aabb = new cgal_aabb;
}

void cgal_aabb_init_from_trimesh(cgal_aabb *aabb,
								 size_t num_points, double (*points)[3],
								 size_t num_faces, size_t (*faces)[3]) {
	for (size_t i = 0; i < num_points; ++i) {
		aabb->mesh.add_vertex(Point(points[i][0], points[i][1], points[i][2]));
	}

	for (size_t i = 0; i < num_faces; ++i) {
		aabb->mesh.add_face(
			Mesh::Vertex_index(faces[i][0]),
			Mesh::Vertex_index(faces[i][1]),
			Mesh::Vertex_index(faces[i][2]));
	}

	auto const mesh_faces = aabb->mesh.faces();

	aabb->tree = Tree(mesh_faces.first, mesh_faces.second, aabb->mesh);
}

void cgal_aabb_dealloc(cgal_aabb **aabb) {
	delete *aabb;
	*aabb = NULL;
}

bool cgal_aabb_test_face_to_face_vis(cgal_aabb const *aabb, size_t i, size_t j) {
	auto const & mesh = aabb->mesh;

    halfedge_descriptor hd = halfedge(Mesh::Face_index(i), mesh);

    Point p_i = CGAL::centroid(mesh.point(source(hd, mesh)),
							   mesh.point(target(hd, mesh)),
							   mesh.point(target(next(hd, mesh), mesh)));

	hd = halfedge(Mesh::Face_index(j), mesh);

	Point p_j = CGAL::centroid(mesh.point(source(hd, mesh)),
							   mesh.point(target(hd, mesh)),
							   mesh.point(target(next(hd, mesh), mesh)));

	auto const d = p_j - p_i;

	Ray ray(p_i, d);

	auto const skip = [i] (size_t k) { return k == i; };

	Ray_intersection intersection = aabb->tree.first_intersection(ray, skip);
	if (!intersection)
		throw std::runtime_error("ray shot from face i to face j escaped!");

	auto const hit_face_index = intersection->second;
	decltype(hit_face_index) target_face_index(j);

	return hit_face_index == target_face_index;
}
