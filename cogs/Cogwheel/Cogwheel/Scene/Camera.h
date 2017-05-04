// Cogwheel scene camera.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_SCENE_CAMERA_H_
#define _COGWHEEL_SCENE_CAMERA_H_

#include <Cogwheel/Core/Renderer.h>
#include <Cogwheel/Math/Conversions.h>
#include <Cogwheel/Math/Matrix.h>
#include <Cogwheel/Math/Ray.h>
#include <Cogwheel/Math/Rect.h>
#include <Cogwheel/Math/Transform.h>
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

    static bool is_allocated() { return m_scene_IDs != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static bool has(Cameras::UID camera_ID) { return m_UID_generator.has(camera_ID); }

    static Cameras::UID create(const std::string& name, SceneRoots::UID scene, 
                               Math::Matrix4x4f projection_matrix, Math::Matrix4x4f inverse_projection_matrix,
                               Core::Renderers::UID renderer_ID = Core::Renderers::UID::invalid_UID());

    static UIDGenerator::ConstIterator begin() { return m_UID_generator.begin(); }
    static UIDGenerator::ConstIterator end() { return m_UID_generator.end(); }
    static Core::Iterable<ConstUIDIterator> get_iterable() { return Core::Iterable<ConstUIDIterator>(begin(), end()); }

    static inline std::string get_name(Cameras::UID camera_ID) { return m_names[camera_ID]; }
    static inline void set_name(Cameras::UID camera_ID, const std::string& name) { m_names[camera_ID] = name; }

    static SceneRoots::UID get_scene_ID(Cameras::UID camera_ID) { return m_scene_IDs[camera_ID]; }

    static Core::Renderers::UID get_renderer_ID(Cameras::UID camera_ID) { return m_renderer_IDs[camera_ID]; }
    static void set_renderer_ID(Cameras::UID camera_ID, Core::Renderers::UID renderer_ID) { m_renderer_IDs[camera_ID] = renderer_ID; }

    static Math::Transform get_transform(Cameras::UID camera_ID) { return m_transforms[camera_ID]; }
    static void set_transform(Cameras::UID camera_ID, Math::Transform transform) { m_transforms[camera_ID] = transform; }
    static Math::Transform get_inverse_view_transform(Cameras::UID camera_ID) { return m_transforms[camera_ID]; }
    static Math::Transform get_view_transform(Cameras::UID camera_ID) { return Math::invert(get_inverse_view_transform(camera_ID)); }

    static Math::Matrix4x4f get_projection_matrix(Cameras::UID camera_ID) { return m_projection_matrices[camera_ID]; }
    static Math::Matrix4x4f get_inverse_projection_matrix(Cameras::UID camera_ID) { return m_inverse_projection_matrices[camera_ID]; }
    static void set_projection_matrices(Cameras::UID camera_ID, Math::Matrix4x4f projection_matrix, Math::Matrix4x4f inv_projection_matrix) {
        m_projection_matrices[camera_ID] = projection_matrix;
        m_inverse_projection_matrices[camera_ID] = inv_projection_matrix;
    }
    static void set_projection_matrices(Cameras::UID camera_ID, Math::Matrix4x4f projection_matrix) {
        set_projection_matrices(camera_ID, projection_matrix, invert(projection_matrix));
    }

    static Math::Matrix4x4f get_view_projection_matrix(Cameras::UID camera_ID) {
        return get_projection_matrix(camera_ID) * to_matrix4x4(get_view_transform(camera_ID));
    }
    static Math::Matrix4x4f get_inverse_view_projection_matrix(Cameras::UID camera_ID) {
        return to_matrix4x4(get_inverse_view_transform(camera_ID)) * get_inverse_projection_matrix(camera_ID);
    }

    static unsigned int get_render_index(Cameras::UID camera_ID) { return m_render_indices[camera_ID]; }
    static void set_render_index(Cameras::UID camera_ID, unsigned int index) { m_render_indices[camera_ID] = index; }
    static Math::Rectf get_viewport(Cameras::UID camera_ID) { return m_viewports[camera_ID]; }
    static void set_viewport(Cameras::UID camera_ID, Math::Rectf projectionport) { m_viewports[camera_ID] = projectionport; }

private:

    static void reserve_camera_data(unsigned int new_capacity, unsigned int old_capacity);

    static UIDGenerator m_UID_generator;

    static std::string* m_names;
    static SceneRoots::UID* m_scene_IDs;
    static Math::Transform* m_transforms;
    static Math::Matrix4x4f* m_projection_matrices;
    static Math::Matrix4x4f* m_inverse_projection_matrices;
    static unsigned int* m_render_indices;
    static Math::Rectf* m_viewports;
    static Core::Renderers::UID* m_renderer_IDs;
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