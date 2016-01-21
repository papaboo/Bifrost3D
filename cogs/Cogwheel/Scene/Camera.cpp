// Cogwheel scene camera.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Scene/Camera.h>

#include <Math/Conversions.h>

using namespace Cogwheel::Math;

namespace Cogwheel {
namespace Scene {

//*****************************************************************************
// Cameras
//*****************************************************************************

Cameras::UIDGenerator Cameras::m_UID_generator = UIDGenerator(0u);

SceneNodes::UID* Cameras::m_parent_IDs = nullptr;
unsigned int* Cameras::m_render_indices = nullptr;
Math::Matrix4x4f* Cameras::m_projection_matrices = nullptr;
Math::Matrix4x4f* Cameras::m_inverse_projection_matrices = nullptr;
Math::Rectf* Cameras::m_viewports = nullptr;

void Cameras::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = UIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_parent_IDs = new SceneNodes::UID[capacity];
    m_render_indices = new unsigned int[capacity];
    m_projection_matrices = new Math::Matrix4x4f[capacity];
    m_inverse_projection_matrices = new Math::Matrix4x4f[capacity];
    m_viewports = new Math::Rectf[capacity];

    // Allocate dummy camera at 0.
    m_parent_IDs[0] = SceneNodes::UID::invalid_UID();
    m_render_indices[0] = 0u;
    m_projection_matrices[0] = Math::Matrix4x4f::zero();
    m_inverse_projection_matrices[0] = Math::Matrix4x4f::zero();
    m_viewports[0] = Math::Rectf(0,0,0,0);
}

void Cameras::deallocate() {
    if (!is_allocated())
        return;

    m_UID_generator = UIDGenerator(0u);

    delete[] m_parent_IDs; m_parent_IDs = nullptr;
    delete[] m_render_indices; m_render_indices = nullptr;
    delete[] m_projection_matrices; m_projection_matrices = nullptr;
    delete[] m_inverse_projection_matrices; m_inverse_projection_matrices = nullptr;
    delete[] m_viewports; m_viewports = nullptr;
}

void Cameras::reserve(unsigned int new_capacity) {
    unsigned int old_capacity = capacity();
    m_UID_generator.reserve(new_capacity);
    reserve_node_data(new_capacity, old_capacity);
}

template <typename T>
static inline T* resize_and_copy_array(T* old_array, unsigned int new_capacity, unsigned int copyable_elements) {
    T* new_array = new T[new_capacity];
    std::copy(old_array, old_array + copyable_elements, new_array);
    delete[] old_array;
    return new_array;
}

void Cameras::reserve_node_data(unsigned int new_capacity, unsigned int old_capacity) {
    const unsigned int copyable_elements = new_capacity < old_capacity ? new_capacity : old_capacity;

    m_parent_IDs = resize_and_copy_array(m_parent_IDs, new_capacity, copyable_elements);

    m_render_indices = resize_and_copy_array(m_render_indices, new_capacity, copyable_elements);
    m_projection_matrices = resize_and_copy_array(m_projection_matrices, new_capacity, copyable_elements);
    m_inverse_projection_matrices = resize_and_copy_array(m_inverse_projection_matrices, new_capacity, copyable_elements);

    m_viewports = resize_and_copy_array(m_viewports, new_capacity, copyable_elements);
}

Cameras::UID Cameras::create(SceneNodes::UID parent_ID, Math::Matrix4x4f projection_matrix, Math::Matrix4x4f inverse_projection_matrix) {
    unsigned int old_capacity = m_UID_generator.capacity();
    UID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_node_data(m_UID_generator.capacity(), old_capacity);

    m_parent_IDs[id] = parent_ID;
    m_render_indices[id] = 0u;
    m_projection_matrices[id] = projection_matrix;
    m_inverse_projection_matrices[id] = inverse_projection_matrix;
    m_viewports[id] = Math::Rectf(0, 0, 1, 1);
    return id;
}

//*****************************************************************************
// Camera Utilities
//*****************************************************************************

namespace CameraUtils {

void compute_perspective_projection(float near_distance, float far_distance, float field_of_view_in_radians, float aspect_ratio,
    Math::Matrix4x4f& projection_matrix, Math::Matrix4x4f& inverse_projection_matrix) {

    // http://www.3dcpptutorials.sk/index.php?id=2, which creates an OpenGL projection matrix (-z forward)
    // Negated the third column to have ?z as forward, see Real-Time Rendering - Third Edition, page 95.
    float f = 1.0f / tan(field_of_view_in_radians * 0.5f);
    float a = (far_distance + near_distance) / (near_distance - far_distance);
    float b = (2.0f * far_distance * near_distance) / (near_distance - far_distance);

    projection_matrix[0][0] = f / aspect_ratio;
    projection_matrix[1][1] = f;
    projection_matrix[2][2] = -a;
    projection_matrix[2][3] = 1.0f;
    projection_matrix[3][2] = b;

    // Yes you could just use inverse_projection_matrix = invert(projection_matrix) as this is by no means performance critical code.
    // But this wasn't done to speed up perspective camera creation. This was done for fun and to have a way to easily derive the inverse perspective matrix later given the perspective matrix.

    const Math::Matrix4x4f& v = projection_matrix;

    inverse_projection_matrix[0][0] = v[1][1] * v[3][2];
    inverse_projection_matrix[0][1] = -0.0f;
    inverse_projection_matrix[0][2] = -0.0f;
    inverse_projection_matrix[0][3] = -0.0f;

    inverse_projection_matrix[1][0] = -0.0f;
    inverse_projection_matrix[1][1] = v[0][0] * v[3][2];
    inverse_projection_matrix[1][2] = -0.0f;
    inverse_projection_matrix[1][3] = -0.0f;

    inverse_projection_matrix[2][0] = -0.0f;
    inverse_projection_matrix[2][1] = -0.0f;
    inverse_projection_matrix[2][2] = -0.0f;
    inverse_projection_matrix[2][3] = v[0][0] * v[1][1];

    inverse_projection_matrix[3][0] = -0.0f;
    inverse_projection_matrix[3][1] = -0.0f;
    inverse_projection_matrix[3][2] = v[0][0] * v[1][1] * v[3][2];
    inverse_projection_matrix[3][3] = - v[0][0] * v[1][1] * v[2][2];

    float determinant = v[0][0] * v[1][1] * v[3][2];
    inverse_projection_matrix /= determinant;
}

Ray ray_from_viewport_point(Cameras::UID camera_ID, Vector2f viewport_point) {
    
    Matrix4x4f inverse_view_matrix = to_matrix4x4(Cameras::get_inverse_view_transform(camera_ID));
    Matrix4x4f& inverse_projection_matrix = Cameras::get_inverse_projection_matrix(camera_ID);
    Matrix4x4f inverse_view_projection_matrix = inverse_projection_matrix * inverse_view_matrix;

    // TODO If I set normalized_screen_pos.z to 0, do I then get the point on the nearplane? Beacuse that is actually the proper ray origin.
    Vector4f normalized_screen_pos = Vector4f(viewport_point.x * 2.0f - 1.0f, viewport_point.y * 2.0f - 1.0f, 1.0f, 1.0f); // TODO We can elliminate some multiplications here by not doing the full mat/vec multiplication.
    Vector4f screenspace_world_pos = normalized_screen_pos * inverse_view_projection_matrix;
    Vector3f point_on_ray = Vector3f(screenspace_world_pos.x, screenspace_world_pos.y, screenspace_world_pos.z) / screenspace_world_pos.w;
    
    SceneNodes::UID camera_node_ID = Cameras::get_parent_ID(camera_ID);
    Transform camera_transform = SceneNodes::get_global_transform(camera_node_ID); // The view transform is the inverse to the camera transform.
    Vector3f ray_origin = camera_transform.translation;

    return Ray(ray_origin, normalize(point_on_ray - ray_origin));
}

} // NS CameraUtils

} // NS Scene
} // NS Cogwheel