#pragma once

#include <stdbool.h>
#include <stddef.h>

typedef struct cgal_aabb cgal_aabb;

void cgal_aabb_alloc(cgal_aabb **aabb);
void cgal_aabb_init_from_trimesh(cgal_aabb *aabb,
								 size_t num_points, double (*points)[3],
								 size_t num_faces, size_t (*faces)[3]);
void cgal_aabb_dealloc(cgal_aabb **aabb);
bool cgal_aabb_test_face_to_face_vis(cgal_aabb const *aabb, size_t i, size_t j);
bool cgal_aabb_ray_from_centroid_is_occluded(cgal_aabb const *aabb, size_t i, double d[3]);
bool cgal_aabb_intersect1(cgal_aabb const *aabb, double const x[3], double const d[3],
						  size_t *i, double xt[3]);
