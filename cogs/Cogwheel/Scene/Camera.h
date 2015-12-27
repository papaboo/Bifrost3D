// Cogwheel scene camera.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_SCENE_CAMERA_H_
#define _COGWHEEL_SCENE_CAMERA_H_

#include <Math/Matrix.h>
#include <Math/Rect.h>
#include <Scene/SceneNode.h>

namespace Cogwheel {
namespace Scene {

class Cameras final {
public:

    typedef Core::TypedUIDGenerator<Cameras> UIDGenerator;
    typedef UIDGenerator::UID UID;

    static bool is_allocated() { return m_parent_IDs != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int capacity);
    static bool has(Cameras::UID cameraID) { return m_UID_generator.has(cameraID); }

    static Cameras::UID create(SceneNodes::UID parent_ID, Math::Matrix4x4f projection_matrix, Math::Matrix4x4f inverse_projection_matrix);

    static SceneNodes::UID get_parent_ID(Cameras::UID camera_ID) { return m_parent_IDs[camera_ID]; }
    static void set_parent_ID(Cameras::UID camera_ID, SceneNodes::UID parent_ID) { m_parent_IDs[camera_ID] = parent_ID; }

    static unsigned int get_render_index(Cameras::UID camera_ID) { return m_render_indices[camera_ID]; }
    static void set_render_index(Cameras::UID camera_ID, unsigned int index) { m_render_indices[camera_ID] = index; }

    static Math::Matrix4x4f get_projection_matrix(Cameras::UID camera_ID) { return m_projection_matrices[camera_ID]; }
    static Math::Matrix4x4f get_inverse_projection_matrix(Cameras::UID camera_ID) { return m_inverse_projection_matrices[camera_ID]; }
    static void set_projection_matrices(Cameras::UID camera_ID, Math::Matrix4x4f projection_matrix, Math::Matrix4x4f inv_projection_matrix) {
        m_projection_matrices[camera_ID] = projection_matrix;
        m_inverse_projection_matrices[camera_ID] = inv_projection_matrix;
    }
    static void set_projection_matrices(Cameras::UID camera_ID, Math::Matrix4x4f projection_matrix) {
        set_projection_matrices(camera_ID, projection_matrix, invert(projection_matrix));
    }

    static Math::Rectf get_viewport(Cameras::UID camera_ID) { return m_viewports[camera_ID]; }
    static void set_viewport(Cameras::UID camera_ID, Math::Rectf projectionport) { m_viewports[camera_ID] = projectionport; }

    // TODO Iterators that iterates through the cameras in order of their render indices.

private:

    static void reserve_node_data(unsigned int capacity, unsigned int oldCapacity);

    static UIDGenerator m_UID_generator;

    static SceneNodes::UID* m_parent_IDs;
    static unsigned int* m_render_indices;
    static Math::Matrix4x4f* m_projection_matrices;
    static Math::Matrix4x4f* m_inverse_projection_matrices;
    static Math::Rectf* m_viewports;
    // TODO Some reference to a backbuffer or render_target to allow cameras to render to windows and FBO's.
};

namespace CameraUtils {

void compute_perspective_projection(float near_distance, float far_distance, float field_of_view_in_radians, float aspect_ratio,
                                    Math::Matrix4x4f& projection_matrix, Math::Matrix4x4f& inverse_projection_matrix);

// TODO compute_orthographic_projection

} // NS CameraUtils

} // NS Scene
} // NS Cogwheel

#endif // _COGWHEEL_SCENE_CAMERA_H_