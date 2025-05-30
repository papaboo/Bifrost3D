// Bifrost scene camera.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_SCENE_CAMERA_H_
#define _BIFROST_SCENE_CAMERA_H_

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Core/Renderer.h>
#include <Bifrost/Math/CameraEffects.h>
#include <Bifrost/Math/Conversions.h>
#include <Bifrost/Math/Matrix.h>
#include <Bifrost/Math/Ray.h>
#include <Bifrost/Math/Rect.h>
#include <Bifrost/Math/Transform.h>
#include <Bifrost/Scene/SceneRoot.h>

#include <functional>

namespace Bifrost {
namespace Scene {

// ------------------------------------------------------------------------------------------------
// Container for screenshots from the camera.
// ------------------------------------------------------------------------------------------------
struct Screenshot {
    enum class Content : unsigned char {
        None = 0u,
        ColorLDR = 1u << 0u,
        ColorHDR = 1u << 1u,
        Depth = 1u << 2u,
        Albedo = 1u << 3u,
        Tint = 1u << 4u,
        Roughness = 1u << 5u,
    };

    Assets::Images::PixelData pixels;
    Content content;
    Assets::PixelFormat format;
    int width, height;

    Screenshot() = default;

    Screenshot(int width, int height, Content content, Assets::PixelFormat format, Assets::Images::PixelData pixels)
        : width(width), height(height), content(content), format(format), pixels(pixels) { }
};

//----------------------------------------------------------------------------
// Camera ID
//----------------------------------------------------------------------------
class Cameras;
typedef Core::TypedUIDGenerator<Cameras> CameraIDGenerator;
typedef CameraIDGenerator::UID CameraID;

// ------------------------------------------------------------------------------------------------
// Container for bifrost matrix cameras.
// ------------------------------------------------------------------------------------------------
class Cameras final {
public:
    using Iterator = CameraIDGenerator::ConstIterator;

    static bool is_allocated() { return m_scene_IDs != nullptr; }
    static void allocate(unsigned int capacity);
    static void deallocate();

    static inline unsigned int capacity() { return m_UID_generator.capacity(); }
    static void reserve(unsigned int new_capacity);
    static inline bool has(CameraID camera_ID) { return m_UID_generator.has(camera_ID) && get_changes(camera_ID).not_set(Change::Destroyed); }

    static CameraID create(const std::string& name, SceneRootID scene_ID,
                           Math::Matrix4x4f projection_matrix, Math::Matrix4x4f inverse_projection_matrix,
                           Core::RendererID renderer_ID = Core::RendererID::invalid_UID());
    static void destroy(CameraID camera_ID);

    static Iterator begin() { return m_UID_generator.begin(); }
    static Iterator end() { return m_UID_generator.end(); }
    static Core::Iterable<Iterator> get_iterable() { return Core::Iterable<Iterator>(begin(), end()); }

    static inline std::string get_name(CameraID camera_ID) { return m_names[camera_ID]; }
    static inline void set_name(CameraID camera_ID, const std::string& name) { m_names[camera_ID] = name; }

    static SceneRootID get_scene_ID(CameraID camera_ID) { return m_scene_IDs[camera_ID]; }

    static Core::RendererID get_renderer_ID(CameraID camera_ID) { return m_renderer_IDs[camera_ID]; }
    static void set_renderer_ID(CameraID camera_ID, Core::RendererID renderer_ID) { 
        m_renderer_IDs[camera_ID] = renderer_ID;
        m_changes.add_change(camera_ID, Change::Renderer);
    }

    static Math::Transform get_transform(CameraID camera_ID) { return m_transforms[camera_ID]; }
    static void set_transform(CameraID camera_ID, Math::Transform transform) { m_transforms[camera_ID] = transform; }
    static Math::Transform get_inverse_view_transform(CameraID camera_ID) { return m_transforms[camera_ID]; }
    static Math::Transform get_view_transform(CameraID camera_ID) { return Math::invert(get_inverse_view_transform(camera_ID)); }

    static Math::Matrix4x4f get_projection_matrix(CameraID camera_ID) { return m_projection_matrices[camera_ID]; }
    static Math::Matrix4x4f get_inverse_projection_matrix(CameraID camera_ID) { return m_inverse_projection_matrices[camera_ID]; }
    static void set_projection_matrices(CameraID camera_ID, Math::Matrix4x4f projection_matrix, Math::Matrix4x4f inv_projection_matrix) {
        m_projection_matrices[camera_ID] = projection_matrix;
        m_inverse_projection_matrices[camera_ID] = inv_projection_matrix;
    }
    static void set_projection_matrices(CameraID camera_ID, Math::Matrix4x4f projection_matrix) {
        set_projection_matrices(camera_ID, projection_matrix, invert(projection_matrix));
    }

    static Math::Matrix4x4f get_view_projection_matrix(CameraID camera_ID) {
        return get_projection_matrix(camera_ID) * to_matrix4x4(get_view_transform(camera_ID));
    }
    static Math::Matrix4x4f get_inverse_view_projection_matrix(CameraID camera_ID) {
        return to_matrix4x4(get_inverse_view_transform(camera_ID)) * get_inverse_projection_matrix(camera_ID);
    }

    // Order that cameras are draw in. Cameras with higher z-index are rendered in front of cameras with lower z-index.
    static int get_z_index(CameraID camera_ID) { return m_z_indices[camera_ID]; }
    static void set_z_index(CameraID camera_ID, int index) { m_z_indices[camera_ID] = index; }
    // Returns an ordered list of camera IDs, where the camera with the lowest Z index is first.
    static std::vector<CameraID> get_z_sorted_IDs();

    static Math::Rectf get_viewport(CameraID camera_ID) { return m_viewports[camera_ID]; }
    static Math::Recti get_window_viewport(CameraID camera_ID, Math::Vector2i window_size);
    static void set_viewport(CameraID camera_ID, Math::Rectf viewport) { m_viewports[camera_ID] = viewport; }

    static Math::CameraEffects::Settings get_effects_settings(CameraID camera_ID) { return m_effects_settings[camera_ID]; }
    static void set_effects_settings(CameraID camera_ID, Math::CameraEffects::Settings settings) { m_effects_settings[camera_ID] = settings; }

    //---------------------------------------------------------------------------------------------
    // Screenshot functionality. Currently only supports getting screenshots based on iteration count.
    //---------------------------------------------------------------------------------------------
    using ScreenshotContent = Core::Bitmask<Screenshot::Content>;
    static void request_screenshot(CameraID camera_ID, ScreenshotContent content_requested, unsigned int minimum_iteration_count = 1) {
        m_screenshot_request[camera_ID].content_requested = content_requested;
        m_screenshot_request[camera_ID].minimum_iteration_count = minimum_iteration_count;
    }
    static bool is_screenshot_requested(CameraID camera_ID) { 
        return m_screenshot_request[camera_ID].content_requested != Screenshot::Content::None;
    }
    static void cancel_screenshot(CameraID camera_ID) { m_screenshot_request[camera_ID].content_requested = Screenshot::Content::None; }
    using ScreenshotFiller = std::function<std::vector<Screenshot>(ScreenshotContent, unsigned int minimum_iteration_count)>;
    // Grabs screenshot data from the renderer and stores it in an intermediate format while the datamodel is considered immutable.
    static void fill_screenshot(CameraID camera_ID, ScreenshotFiller screenshot_filler);
    static ScreenshotContent pending_screenshots(CameraID camera_ID);
    static Assets::Image resolve_screenshot(CameraID camera_ID, Screenshot::Content image_content, const std::string& name); // Resolves the last screenshot into an image.

    //---------------------------------------------------------------------------------------------
    // Changes since last game loop tick.
    //---------------------------------------------------------------------------------------------
    enum class Change : unsigned char {
        None = 0u,
        Created = 1u << 0u,
        Destroyed = 1u << 1u,
        Renderer = 1u << 2u,
        All = Created | Destroyed | Renderer
    };
    typedef Core::Bitmask<Change> Changes;

    static inline Changes get_changes(CameraID node_ID) { return m_changes.get_changes(node_ID); }

    typedef std::vector<CameraID>::iterator ChangedIterator;
    static Core::Iterable<ChangedIterator> get_changed_cameras() { return m_changes.get_changed_resources(); }

    static void reset_change_notifications();

private:

    static void reserve_camera_data(unsigned int new_capacity, unsigned int old_capacity);

    static CameraIDGenerator m_UID_generator;

    static std::string* m_names;
    static SceneRootID* m_scene_IDs;
    static Math::Transform* m_transforms;
    static Math::Matrix4x4f* m_projection_matrices;
    static Math::Matrix4x4f* m_inverse_projection_matrices;
    static int* m_z_indices;
    static Math::Rectf* m_viewports;
    static Core::RendererID* m_renderer_IDs;
    static Math::CameraEffects::Settings* m_effects_settings;

    struct ScreenshotRequest {
        Core::Bitmask<Screenshot::Content> content_requested;
        unsigned int minimum_iteration_count;

        std::vector<Screenshot> images;
    };

    static ScreenshotRequest* m_screenshot_request;

    static Core::ChangeSet<Changes, CameraID> m_changes;
};

//-------------------------------------------------------------------------------------------------
// Camera utils
//-------------------------------------------------------------------------------------------------
namespace CameraUtils {

void compute_perspective_projection(float near_distance, float far_distance, float field_of_view_in_radians, float aspect_ratio,
                                    Math::Matrix4x4f& projection_matrix, Math::Matrix4x4f& inverse_projection_matrix);

void compute_orthographic_projection(float width, float height, float depth,
                                     Math::Matrix4x4f& projection_matrix, Math::Matrix4x4f& inverse_projection_matrix);

// Future work: compute_orthographic_projection

// Generates a ray in worldspace shot from the point in the viewport of the referenced camera.
// The viewport point is in normalized coordinates [0, 1]^2.
Math::Ray ray_from_viewport_point(CameraID camera_ID, Math::Vector2f viewport_point);

} // NS CameraUtils

} // NS Scene
} // NS Bifrost

#endif // _BIFROST_SCENE_CAMERA_H_
