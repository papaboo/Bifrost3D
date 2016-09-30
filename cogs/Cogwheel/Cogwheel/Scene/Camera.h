// Cogwheel scene camera.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_SCENE_CAMERA_H_
#define _COGWHEEL_SCENE_CAMERA_H_

#include <Cogwheel/Math/Matrix.h>
#include <Cogwheel/Math/Ray.h>
#include <Cogwheel/Math/Rect.h>
#include <Cogwheel/Scene/SceneNode.h>
#include <Cogwheel/Scene/SceneRoot.h>

namespace Cogwheel {
namespace Scene {

// ---------------------------------------------------------------------------
// Container for cogwheel matrix cameras.
// Future work
// * Iterators that iterates through the cameras in order of their render indices.
// * Reference a backbuffer or render_target to allow cameras to render to windows and FBO's.
// * Change flags.
// ---------------------------------------------------------------------------
class Cameras final {
public:
    typedef Core::TypedUIDGenerator<Cameras> UIDGenerator;
    typedef UIDGenerator::UID UID;
    typedef UIDGenerator::ConstIterator ConstUIDIterator;

    static bool is_allocated() { return m_node_IDs != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static bool has(Cameras::UID cameraID) { return m_UID_generator.has(cameraID); }

    static Cameras::UID create(SceneNodes::UID parent_ID, SceneRoots::UID scene, Math::Matrix4x4f projection_matrix, Math::Matrix4x4f inverse_projection_matrix);

    static UIDGenerator::ConstIterator begin() { return m_UID_generator.begin(); }
    static UIDGenerator::ConstIterator end() { return m_UID_generator.end(); }
    static Core::Iterable<ConstUIDIterator> get_iterable() { return Core::Iterable<ConstUIDIterator>(begin(), end()); }

    static SceneNodes::UID get_node_ID(Cameras::UID camera_ID) { return m_node_IDs[camera_ID]; }
    static void set_node_ID(Cameras::UID camera_ID, SceneNodes::UID node_ID) { m_node_IDs[camera_ID] = node_ID; }

    static SceneRoots::UID get_scene_ID(Cameras::UID camera_ID) { return m_scene_IDs[camera_ID]; }

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

    static Math::Transform get_inverse_view_transform(Cameras::UID camera_ID) { return SceneNodes::get_global_transform(m_node_IDs[camera_ID]); }
    static Math::Transform get_view_transform(Cameras::UID camera_ID) { return Math::invert(get_inverse_view_transform(camera_ID)); }

    static Math::Rectf get_viewport(Cameras::UID camera_ID) { return m_viewports[camera_ID]; }
    static void set_viewport(Cameras::UID camera_ID, Math::Rectf projectionport) { m_viewports[camera_ID] = projectionport; }

private:

    static void reserve_camera_data(unsigned int new_capacity, unsigned int old_capacity);

    static UIDGenerator m_UID_generator;

    static SceneNodes::UID* m_node_IDs;
    static SceneRoots::UID* m_scene_IDs;
    static unsigned int* m_render_indices;
    static Math::Matrix4x4f* m_projection_matrices;
    static Math::Matrix4x4f* m_inverse_projection_matrices;
    static Math::Rectf* m_viewports;
};

namespace CameraUtils {

void compute_perspective_projection(float near_distance, float far_distance, float field_of_view_in_radians, float aspect_ratio,
                                    Math::Matrix4x4f& projection_matrix, Math::Matrix4x4f& inverse_projection_matrix);

// Future work: compute_orthographic_projection

Math::Ray ray_from_viewport_point(Cameras::UID camera_ID, Math::Vector2f viewport_point);

} // NS CameraUtils

} // NS Scene
} // NS Cogwheel

#endif // _COGWHEEL_SCENE_CAMERA_H_