// Bifrost scene camera.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Bifrost/Scene/Camera.h>

#include <assert.h>

using namespace Bifrost::Math;

namespace Bifrost {
namespace Scene {

//*****************************************************************************
// Cameras
//*****************************************************************************

CameraIDGenerator Cameras::m_UID_generator = CameraIDGenerator(0u);

std::string* Cameras::m_names = nullptr;
SceneRootID* Cameras::m_scene_IDs = nullptr;
int* Cameras::m_z_indices = nullptr;
Transform* Cameras::m_transforms = nullptr;
Matrix4x4f* Cameras::m_projection_matrices = nullptr;
Matrix4x4f* Cameras::m_inverse_projection_matrices = nullptr;
Rectf* Cameras::m_viewports = nullptr;
Core::RendererID* Cameras::m_renderer_IDs = nullptr;
CameraEffects::Settings* Cameras::m_effects_settings = nullptr;
Cameras::ScreenshotRequest* Cameras::m_screenshot_request = nullptr;
Core::ChangeSet<Cameras::Changes, CameraID> Cameras::m_changes;

void Cameras::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = CameraIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_names = new std::string[capacity];
    m_scene_IDs = new SceneRootID[capacity];
    m_transforms = new Transform[capacity];
    m_projection_matrices = new Matrix4x4f[capacity];
    m_inverse_projection_matrices = new Matrix4x4f[capacity];
    m_z_indices = new int[capacity];
    m_viewports = new Rectf[capacity];
    m_renderer_IDs = new Core::RendererID[capacity];
    m_effects_settings = new CameraEffects::Settings[capacity];
    m_screenshot_request = new ScreenshotRequest[capacity];
    m_changes = Core::ChangeSet<Changes, CameraID>(capacity);

    // Allocate dummy camera at 0.
    m_names[0] = "Dummy camera";
    m_transforms[0] = Transform::identity();
    m_scene_IDs[0] = SceneRootID::invalid_UID();
    m_z_indices[0] = 0;
    m_projection_matrices[0] = Matrix4x4f::zero();
    m_inverse_projection_matrices[0] = Matrix4x4f::zero();
    m_viewports[0] = Rectf(0,0,0,0);
    m_renderer_IDs[0] = Core::RendererID::invalid_UID();
    m_effects_settings[0] = {};
    m_screenshot_request[0] = {};
}

void Cameras::deallocate() {
    if (!is_allocated())
        return;

    m_UID_generator = CameraIDGenerator(0u);

    delete[] m_names; m_names = nullptr;
    delete[] m_scene_IDs; m_scene_IDs = nullptr;
    delete[] m_transforms; m_transforms = nullptr;
    delete[] m_projection_matrices; m_projection_matrices = nullptr;
    delete[] m_inverse_projection_matrices; m_inverse_projection_matrices = nullptr;
    delete[] m_z_indices; m_z_indices = nullptr;
    delete[] m_viewports; m_viewports = nullptr;
    delete[] m_renderer_IDs; m_renderer_IDs = nullptr;
    delete[] m_effects_settings; m_effects_settings = nullptr;
    delete[] m_screenshot_request; m_screenshot_request = nullptr;

    m_changes.resize(0);
}

void Cameras::reserve(unsigned int new_capacity) {
    unsigned int old_capacity = capacity();
    m_UID_generator.reserve(new_capacity);
    reserve_camera_data(m_UID_generator.capacity(), old_capacity);
}

template <typename T>
static inline T* resize_and_copy_array(T* old_array, unsigned int new_capacity, unsigned int copyable_elements) {
    T* new_array = new T[new_capacity];
    std::copy(old_array, old_array + copyable_elements, new_array);
    delete[] old_array;
    return new_array;
}

void Cameras::reserve_camera_data(unsigned int new_capacity, unsigned int old_capacity) {
    assert(m_scene_IDs != nullptr);
    assert(m_transforms != nullptr);
    assert(m_projection_matrices != nullptr);
    assert(m_inverse_projection_matrices != nullptr);
    assert(m_z_indices != nullptr);
    assert(m_viewports != nullptr);

    const unsigned int copyable_elements = new_capacity < old_capacity ? new_capacity : old_capacity;

    m_names = resize_and_copy_array(m_names, new_capacity, copyable_elements);

    m_scene_IDs = resize_and_copy_array(m_scene_IDs, new_capacity, copyable_elements);

    m_transforms = resize_and_copy_array(m_transforms, new_capacity, copyable_elements);
    m_projection_matrices = resize_and_copy_array(m_projection_matrices, new_capacity, copyable_elements);
    m_inverse_projection_matrices = resize_and_copy_array(m_inverse_projection_matrices, new_capacity, copyable_elements);

    m_z_indices = resize_and_copy_array(m_z_indices, new_capacity, copyable_elements);
    m_viewports = resize_and_copy_array(m_viewports, new_capacity, copyable_elements);
    m_renderer_IDs = resize_and_copy_array(m_renderer_IDs, new_capacity, copyable_elements);
    m_effects_settings = resize_and_copy_array(m_effects_settings, new_capacity, copyable_elements);
    m_screenshot_request = resize_and_copy_array(m_screenshot_request, new_capacity, copyable_elements);

    m_changes.resize(new_capacity);
}

CameraID Cameras::create(const std::string& name, SceneRootID scene_ID, 
                         Matrix4x4f projection_matrix, Matrix4x4f inverse_projection_matrix, 
                         Core::RendererID renderer_ID) {
    assert(m_names != nullptr);
    assert(m_scene_IDs != nullptr);
    assert(m_z_indices != nullptr);
    assert(m_transforms != nullptr);
    assert(m_projection_matrices != nullptr);
    assert(m_inverse_projection_matrices != nullptr);
    assert(m_viewports != nullptr);

    if (!SceneRoots::has(scene_ID))
        return CameraID::invalid_UID();

    unsigned int old_capacity = m_UID_generator.capacity();
    CameraID id = m_UID_generator.generate();
    if (old_capacity != m_UID_generator.capacity())
        // The capacity has changed and the size of all arrays need to be adjusted.
        reserve_camera_data(m_UID_generator.capacity(), old_capacity);

    m_names[id] = name;
    m_scene_IDs[id] = scene_ID;
    m_z_indices[id] = 0;
    m_transforms[id] = Math::Transform::identity();
    m_projection_matrices[id] = projection_matrix;
    m_inverse_projection_matrices[id] = inverse_projection_matrix;
    m_viewports[id] = Rectf(0, 0, 1, 1);
    m_renderer_IDs[id] = Core::Renderers::has(renderer_ID) ? renderer_ID : *Core::Renderers::begin();
    m_effects_settings[id] = CameraEffects::Settings::default();
    m_screenshot_request[id] = {};
    m_changes.set_change(id, Change::Created);

    return id;
}

void Cameras::destroy(CameraID camera_ID) {
    // We don't actually destroy anything when destroying a camera.
    // The properties will get overwritten later when a new camera is created in same the spot.
    if (has(camera_ID))
        m_changes.add_change(camera_ID, Change::Destroyed);
}

std::vector<CameraID> Cameras::get_z_sorted_IDs() {
    auto IDs = std::vector<CameraID>();
    IDs.reserve(capacity());
    for (CameraID camera_ID : get_iterable())
        IDs.push_back(camera_ID);
    std::sort(IDs.begin(), IDs.end(), [](CameraID lhs, CameraID rhs) { return Cameras::get_z_index(lhs) < Cameras::get_z_index(rhs); });
    return IDs;
}

Recti Cameras::get_window_viewport(CameraID camera_ID, Vector2i window_size) {
    Rectf viewportf = Cameras::get_viewport(camera_ID);
    Recti window_viewport;
    window_viewport.x = int(round(viewportf.x * window_size.x));
    window_viewport.width = int(round(viewportf.width * window_size.x));
    window_viewport.y = int(round(viewportf.y * window_size.y));
    window_viewport.height = int(round(viewportf.height * window_size.y));
    return window_viewport;
}

void Cameras::fill_screenshot(CameraID camera_ID, ScreenshotFiller screenshot_filler) {
    if (is_screenshot_requested(camera_ID)) {
        auto& screenshot_info = m_screenshot_request[camera_ID];
        auto screenshots = screenshot_filler(screenshot_info.content_requested, screenshot_info.minimum_iteration_count);
        for (auto image : screenshots) {
            assert(screenshot_info.content_requested.is_set(image.content));
            screenshot_info.images.push_back(image);
            screenshot_info.content_requested ^= image.content;
        }
    }
}

Cameras::ScreenshotContent Cameras::pending_screenshots(CameraID camera_ID) {
    ScreenshotContent content = Screenshot::Content::None;
    for (auto& image : m_screenshot_request[camera_ID].images)
        content |= image.content;
    return content;
}

Assets::Image Cameras::resolve_screenshot(CameraID camera_ID, Screenshot::Content image_content, const std::string& name) {
    auto& info = m_screenshot_request[camera_ID];
    for (int i = 0; i < info.images.size(); ++i) {
        auto& screen_shot = info.images[i];
        if (screen_shot.content == image_content) {
            bool is_HDR = screen_shot.content == Screenshot::Content::ColorHDR;
            auto image = Assets::Image::create2D(name, screen_shot.format, !is_HDR, Vector2ui(screen_shot.width, screen_shot.height), screen_shot.pixels);
            info.images.erase(info.images.begin() + i);
            return image;
        }
    }

    return Assets::Image::invalid();
}

void Cameras::reset_change_notifications() {
    for (CameraID camera_ID : get_changed_cameras())
        if (get_changes(camera_ID).is_set(Change::Destroyed))
            m_UID_generator.erase(camera_ID);
    m_changes.reset_change_notifications();
}

//*****************************************************************************
// Camera Utilities
//*****************************************************************************

namespace CameraUtils {

void compute_perspective_projection(float near_distance, float far_distance, float field_of_view_in_radians, float aspect_ratio,
                                    Matrix4x4f& projection_matrix, Matrix4x4f& inverse_projection_matrix) {

    // http://www.3dcpptutorials.sk/index.php?id=2, which creates an OpenGL projection matrix (-Z forward)
    // Negated the third column to have +Z as forward, see Real-Time Rendering - Third Edition, page 95.
    float f = 1.0f / tan(field_of_view_in_radians * 0.5f);
    float a = (far_distance + near_distance) / (near_distance - far_distance);
    float b = (2.0f * far_distance * near_distance) / (near_distance - far_distance);

    projection_matrix = {};
    projection_matrix[0][0] = f / aspect_ratio;
    projection_matrix[0][1] = projection_matrix[0][2] = projection_matrix[0][3] = 0.0f;
    projection_matrix[1][1] = f;
    projection_matrix[1][0] = projection_matrix[1][2] = projection_matrix[1][3] = 0.0f;
    projection_matrix[2][0] = projection_matrix[2][1] = 0.0f;
    projection_matrix[2][2] = -a;
    projection_matrix[2][3] = b;
    projection_matrix[3][0] = projection_matrix[3][1] = projection_matrix[3][3] = 0.0f;
    projection_matrix[3][2] = 1.0f;

    // Yes you could just use inverse_projection_matrix = invert(projection_matrix) as this is by no means performance critical code.
    // But this wasn't done to speed up perspective camera creation. This was done to have a way to easily derive the inverse perspective matrix later given the perspective matrix.

    inverse_projection_matrix = {};
    inverse_projection_matrix[0][0] = 1.0f / projection_matrix[0][0];
    inverse_projection_matrix[1][1] = 1.0f / projection_matrix[1][1];
    inverse_projection_matrix[2][3] = 1.0f;
    inverse_projection_matrix[3][2] = 1.0f / projection_matrix[2][3];
    inverse_projection_matrix[3][3] = - projection_matrix[2][2] / projection_matrix[2][3];
}

void compute_orthographic_projection(float width, float height, float depth,
                                     Matrix4x4f& projection_matrix, Matrix4x4f& inverse_projection_matrix) {
    // Based on 
    // https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/orthographic-projection-matrix.html
    // Simplified as the projection's width and height are centered around 0,0 and the near plane is located at 0.
    projection_matrix = {};
    projection_matrix[0][0] = 2 / width;
    projection_matrix[1][1] = 2 / height;
    projection_matrix[2][2] = 2 / depth;
    projection_matrix[2][3] = -1;
    projection_matrix[3][3] = 1;

    // Inline computation of inverse matrix. Has higher precision than the general invert and is faster.
    inverse_projection_matrix = {};
    inverse_projection_matrix[0][0] = 0.5f * width;
    inverse_projection_matrix[1][1] = 0.5f * height;
    inverse_projection_matrix[2][2] = 0.5f * depth;
    inverse_projection_matrix[2][3] = 0.5f * depth;
    inverse_projection_matrix[3][3] = 1.0f;
}

Ray ray_from_viewport_point(CameraID camera_ID, Vector2f viewport_point) {

    Matrix4x4f inverse_projection_matrix = Cameras::get_inverse_view_projection_matrix(camera_ID);
    Vector4f NDC_near_pos = Vector4f(viewport_point.x * 2.0f - 1.0f, viewport_point.y * 2.0f - 1.0f, -1.0f, 1.0f);
    Vector4f scaled_near_world_pos = inverse_projection_matrix * NDC_near_pos;
    Vector3f ray_to_near_plane = Vector3f(scaled_near_world_pos.x, scaled_near_world_pos.y, scaled_near_world_pos.z) / scaled_near_world_pos.w;

    Vector4f scaled_far_world_pos = scaled_near_world_pos + 2.0f * inverse_projection_matrix.get_column(2);
    Vector3f ray_to_far_plane = Vector3f(scaled_far_world_pos.x, scaled_far_world_pos.y, scaled_far_world_pos.z) / scaled_far_world_pos.w;

    return Ray(ray_to_near_plane, normalize(ray_to_far_plane - ray_to_near_plane));
}

} // NS CameraUtils

} // NS Scene
} // NS Bifrost
