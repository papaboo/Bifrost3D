// Bifrost scene camera.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Bifrost/Scene/Camera.h>

#include <Bifrost/Math/Conversions.h>

#include <assert.h>

using namespace Bifrost::Math;

namespace Bifrost {
namespace Scene {

//*****************************************************************************
// Cameras
//*****************************************************************************

Cameras::UIDGenerator Cameras::m_UID_generator = UIDGenerator(0u);

std::string* Cameras::m_names = nullptr;
SceneRoots::UID* Cameras::m_scene_IDs = nullptr;
int* Cameras::m_z_indices = nullptr;
Transform* Cameras::m_transforms = nullptr;
Matrix4x4f* Cameras::m_projection_matrices = nullptr;
Matrix4x4f* Cameras::m_inverse_projection_matrices = nullptr;
Rectf* Cameras::m_viewports = nullptr;
Core::Renderers::UID* Cameras::m_renderer_IDs = nullptr;
CameraEffects::Settings* Cameras::m_effects_settings = nullptr;
Cameras::ScreenshotInfo* Cameras::m_screenshot_info = nullptr;
Core::ChangeSet<Cameras::Changes, Cameras::UID> Cameras::m_changes;

void Cameras::allocate(unsigned int capacity) {
    if (is_allocated())
        return;

    m_UID_generator = UIDGenerator(capacity);
    capacity = m_UID_generator.capacity();

    m_names = new std::string[capacity];
    m_scene_IDs = new SceneRoots::UID[capacity];
    m_transforms = new Transform[capacity];
    m_projection_matrices = new Matrix4x4f[capacity];
    m_inverse_projection_matrices = new Matrix4x4f[capacity];
    m_z_indices = new int[capacity];
    m_viewports = new Rectf[capacity];
    m_renderer_IDs = new Core::Renderers::UID[capacity];
    m_effects_settings = new CameraEffects::Settings[capacity];
    m_screenshot_info = new ScreenshotInfo[capacity];
    m_changes = Core::ChangeSet<Changes, UID>(capacity);

    // Allocate dummy camera at 0.
    m_names[0] = "Dummy camera";
    m_transforms[0] = Transform::identity();
    m_scene_IDs[0] = SceneRoots::UID::invalid_UID();
    m_z_indices[0] = 0;
    m_projection_matrices[0] = Matrix4x4f::zero();
    m_inverse_projection_matrices[0] = Matrix4x4f::zero();
    m_viewports[0] = Rectf(0,0,0,0);
    m_renderer_IDs[0] = Core::Renderers::UID::invalid_UID();
    m_effects_settings[0] = {};
    m_screenshot_info[0] = {};
}

void Cameras::deallocate() {
    if (!is_allocated())
        return;

    m_UID_generator = UIDGenerator(0u);

    delete[] m_names; m_names = nullptr;
    delete[] m_scene_IDs; m_scene_IDs = nullptr;
    delete[] m_transforms; m_transforms = nullptr;
    delete[] m_projection_matrices; m_projection_matrices = nullptr;
    delete[] m_inverse_projection_matrices; m_inverse_projection_matrices = nullptr;
    delete[] m_z_indices; m_z_indices = nullptr;
    delete[] m_viewports; m_viewports = nullptr;
    delete[] m_renderer_IDs; m_renderer_IDs = nullptr;
    delete[] m_effects_settings; m_effects_settings = nullptr;
    delete[] m_screenshot_info; m_screenshot_info = nullptr;

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
    m_screenshot_info = resize_and_copy_array(m_screenshot_info, new_capacity, copyable_elements);

    m_changes.resize(new_capacity);
}

Cameras::UID Cameras::create(const std::string& name, SceneRoots::UID scene_ID, 
                             Matrix4x4f projection_matrix, Matrix4x4f inverse_projection_matrix, 
                             Core::Renderers::UID renderer_ID) {
    assert(m_names != nullptr);
    assert(m_scene_IDs != nullptr);
    assert(m_z_indices != nullptr);
    assert(m_transforms != nullptr);
    assert(m_projection_matrices != nullptr);
    assert(m_inverse_projection_matrices != nullptr);
    assert(m_viewports != nullptr);

    if (!SceneRoots::has(scene_ID))
        return Cameras::UID::invalid_UID();

    unsigned int old_capacity = m_UID_generator.capacity();
    UID id = m_UID_generator.generate();
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
    m_screenshot_info[id] = {};
    m_changes.set_change(id, Change::Created);

    return id;
}

void Cameras::destroy(Cameras::UID camera_ID) {
    // We don't actually destroy anything when destroying a camera.
    // The properties will get overwritten later when a new camera is created in same the spot.
    if (m_UID_generator.erase(camera_ID))
        m_changes.set_change(camera_ID, Change::Destroyed);
}

std::vector<Cameras::UID> Cameras::get_z_sorted_IDs() {
    auto IDs = std::vector<Cameras::UID>();
    IDs.reserve(capacity());
    for (Cameras::UID camera_ID : get_iterable())
        IDs.push_back(camera_ID);
    std::sort(IDs.begin(), IDs.end(), [](UID lhs, UID rhs) { return Cameras::get_z_index(lhs) < Cameras::get_z_index(rhs); });
    return IDs;
}

void Cameras::fill_screenshot(Cameras::UID camera_ID, ScreenshotFiller screenshot_filler) {
    if (is_screenshot_requested(camera_ID)) {
        auto& screenshot_info = m_screenshot_info[camera_ID];
        Assets::Images::PixelData pixels = screenshot_filler(screenshot_info.pixel_format, screenshot_info.minimum_iteration_count, screenshot_info.width, screenshot_info.height);
        if (pixels != nullptr) {
            if (screenshot_info.pixels != nullptr)
                delete[] screenshot_info.pixels;
            screenshot_info.pixels = pixels;
        }
    }
}

Assets::Images::UID Cameras::resolve_screenshot(Cameras::UID camera_ID, const std::string& name) {
    auto& info = m_screenshot_info[camera_ID];
    if (info.pixels == nullptr)
        return Assets::Images::UID::invalid_UID();

    bool is_HDR = info.pixel_format == Assets::PixelFormat::RGBA_Float;
    auto image_ID = Assets::Images::create2D(name, info.pixel_format, is_HDR ? 1.0f : 2.2f, Vector2ui(info.width, info.height), info.pixels);
    info = {};
    return image_ID;
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
    // But this wasn't done to speed up perspective camera creation. This was done for fun and to have a way to easily derive the inverse perspective matrix later given the perspective matrix.
    
    const Matrix4x4f& v = projection_matrix;
    float determinant = v[0][0] * v[1][1] * v[2][3];

    inverse_projection_matrix[0][0] = v[1][1] * v[2][3] / determinant;
    inverse_projection_matrix[0][1] = 0.0f;
    inverse_projection_matrix[0][2] = 0.0f;
    inverse_projection_matrix[0][3] = 0.0f;

    inverse_projection_matrix[1][0] = 0.0f;
    inverse_projection_matrix[1][1] = v[0][0] * v[2][3] / determinant;
    inverse_projection_matrix[1][2] = 0.0f;
    inverse_projection_matrix[1][3] = 0.0f;

    inverse_projection_matrix[2][0] = 0.0f;
    inverse_projection_matrix[2][1] = 0.0f;
    inverse_projection_matrix[2][2] = 0.0f;
    inverse_projection_matrix[2][3] = 1.0f;

    inverse_projection_matrix[3][0] = 0.0f;
    inverse_projection_matrix[3][1] = 0.0f;
    inverse_projection_matrix[3][2] = v[0][0] * v[1][1] / determinant;
    inverse_projection_matrix[3][3] = - v[0][0] * v[1][1] * v[2][2] / determinant;
}

Ray ray_from_viewport_point(Cameras::UID camera_ID, Vector2f viewport_point) {
    
    Matrix4x4f inverse_view_matrix = to_matrix4x4(Cameras::get_inverse_view_transform(camera_ID));
    Matrix4x4f inverse_projection_matrix = Cameras::get_inverse_projection_matrix(camera_ID);
    Matrix4x4f inverse_view_projection_matrix = inverse_view_matrix * inverse_projection_matrix;

    // NOTE We can elliminate some multiplications here by not doing the full mat/vec multiplication or does the compiler already do that for us (constants and all that.)
    Vector4f normalized_projected_pos = Vector4f(viewport_point.x * 2.0f - 1.0f, viewport_point.y * 2.0f - 1.0f, -1.0f, 1.0f);
    Vector4f projected_world_pos = inverse_view_projection_matrix * normalized_projected_pos;
    Vector3f ray_origin = Vector3f(projected_world_pos.x, projected_world_pos.y, projected_world_pos.z) / projected_world_pos.w;
    
    Vector3f camera_position = Cameras::get_transform(camera_ID).translation;

    return Ray(ray_origin, normalize(ray_origin - camera_position));
}

} // NS CameraUtils

} // NS Scene
} // NS Bifrost
