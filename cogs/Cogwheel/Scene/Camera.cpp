// Cogwheel scene camera.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Scene/Camera.h>

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
    m_parent_IDs[0] = SceneNodes::UID::InvalidUID();
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
static inline T* resize_and_copy_array(T* oldArray, unsigned int newCapacity, unsigned int copyableElements) {
    T* newArray = new T[newCapacity];
    std::copy(oldArray, oldArray + copyableElements, newArray);
    delete[] oldArray;
    return newArray;
}

void Cameras::reserve_node_data(unsigned int newCapacity, unsigned int oldCapacity) {
    const unsigned int copyableElements = newCapacity < oldCapacity ? newCapacity : oldCapacity;

    m_parent_IDs = resize_and_copy_array(m_parent_IDs, newCapacity, copyableElements);

    m_render_indices = resize_and_copy_array(m_render_indices, newCapacity, copyableElements);
    m_projection_matrices = resize_and_copy_array(m_projection_matrices, newCapacity, copyableElements);
    m_inverse_projection_matrices = resize_and_copy_array(m_inverse_projection_matrices, newCapacity, copyableElements);

    m_viewports = resize_and_copy_array(m_viewports, newCapacity, copyableElements);
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

    // http://www.3dcpptutorials.sk/index.php?id=2
    float f = 1.0f / tan(field_of_view_in_radians * 0.5f);
    float a = (far_distance + near_distance) / (near_distance - far_distance);
    float b = (2.0f * far_distance * near_distance) / (near_distance - far_distance);

    projection_matrix[0][0] = f / aspect_ratio;
    projection_matrix[1][1] = f;
    projection_matrix[2][2] = a;
    projection_matrix[2][3] = -1.0f;
    projection_matrix[3][2] = b;

    // Yes you could just use inverse_projection_matrix = invert(projection_matrix) as this is by no means performance critical code.
    // But this wasn't done to speed up perspective camera creation. This was done for fun and to have a way to easily derive the inverse perspective matrix later given the perspective matrix.

    Math::Matrix4x4f& v = projection_matrix;

    inverse_projection_matrix[0][0] = v[1][1] * v[3][2];
    inverse_projection_matrix[0][1] = 0.0f;
    inverse_projection_matrix[0][2] = 0.0f;
    inverse_projection_matrix[0][3] = 0.0f;

    inverse_projection_matrix[1][0] = 0.0f;
    inverse_projection_matrix[1][1] = v[0][0] * v[3][2];
    inverse_projection_matrix[1][2] = 0.0f;
    inverse_projection_matrix[1][3] = 0.0f;

    inverse_projection_matrix[2][0] = 0.0f;
    inverse_projection_matrix[2][1] = 0.0f;
    inverse_projection_matrix[2][2] = 0.0f;
    inverse_projection_matrix[2][3] = v[0][0] * v[1][1];

    inverse_projection_matrix[3][0] = 0.0f;
    inverse_projection_matrix[3][1] = 0.0f;
    inverse_projection_matrix[3][2] = - v[0][0] * v[1][1] * v[3][2];
    inverse_projection_matrix[3][3] = v[0][0] * v[1][1] * v[2][2];

    float determinant = v[0][0] * v[1][1] * v[3][2];
    inverse_projection_matrix /= determinant;
}

} // NS CameraUtils

} // NS Scene
} // NS Cogwheel