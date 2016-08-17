#include <CornellBoxScene.h>
#include <MaterialScene.h>
#include <SphereScene.h>
#include <TestScene.h>

#include <Cogwheel/Assets/Mesh.h>
#include <Cogwheel/Assets/MeshModel.h>
#include <Cogwheel/Core/Engine.h>
#include <Cogwheel/Input/Keyboard.h>
#include <Cogwheel/Input/Mouse.h>
#include <Cogwheel/Scene/Camera.h>
#include <Cogwheel/Scene/LightSource.h>
#include <Cogwheel/Scene/SceneNode.h>
#include <Cogwheel/Scene/SceneRoot.h>

#include <GLFWDriver.h>

#include <ObjLoader/ObjLoader.h>
#include <StbImageLoader/StbImageLoader.h>

#include <OptiXRenderer/Renderer.h>

#include <cstdio>
#include <iostream>

using namespace Cogwheel::Assets;
using namespace Cogwheel::Core;
using namespace Cogwheel::Input;
using namespace Cogwheel::Math;
using namespace Cogwheel::Scene;

static std::string g_scene;
static std::string g_environment;
static RGB g_environment_color = RGB(0.68f, 0.92f, 1.0f);
static float g_scene_size;

class Navigation final {
public:

    Navigation(SceneNodes::UID node_ID, float velocity) 
        : m_node_ID(node_ID)
        , m_vertical_rotation(0.0f) 
        , m_horizontal_rotation(0.0f)
        , m_velocity(velocity)
    { }

    void navigate(Engine& engine) {
        const Keyboard* keyboard = engine.get_keyboard();
        const Mouse* mouse = engine.get_mouse();

        SceneNode node = m_node_ID;
        Transform transform = node.get_global_transform();

        { // Translation
            float strafing = 0.0f;
            if (keyboard->is_pressed(Keyboard::Key::D))
                strafing = 1.0f;
            if (keyboard->is_pressed(Keyboard::Key::A))
                strafing -= 1.0f;

            float forward = 0.0f;
            if (keyboard->is_pressed(Keyboard::Key::W))
                forward = 1.0f;
            if (keyboard->is_pressed(Keyboard::Key::S))
                forward -= 1.0f;

            float velocity = m_velocity;
            if (keyboard->is_pressed(Keyboard::Key::LeftShift) || keyboard->is_pressed(Keyboard::Key::RightShift))
                velocity *= 5.0f;

            if (strafing != 0.0f || forward != 0.0f) {
                Vector3f translation_offset = transform.rotation * Vector3f(strafing, 0.0f, forward);
                float dt = engine.get_time().is_paused() ? engine.get_time().get_raw_delta_time() : engine.get_time().get_smooth_delta_time();
                transform.translation += normalize(translation_offset) * velocity * dt;
            }
        }

        { // Rotation
            if (mouse->is_pressed(Mouse::Button::Left)) {

                m_vertical_rotation += degrees_to_radians(float(mouse->get_delta().x));

                // Clamp horizontal rotation to -89 and 89 degrees to avoid turning the camera on it's head and the singularities of cross products at the poles.
                m_horizontal_rotation += degrees_to_radians(float(mouse->get_delta().y));
                m_horizontal_rotation = clamp(m_horizontal_rotation, -PI<float>() * 0.49f, PI<float>() * 0.49f);

                transform.rotation = Quaternionf::from_angle_axis(m_vertical_rotation, Vector3f::up()) * Quaternionf::from_angle_axis(m_horizontal_rotation, Vector3f::right());
            }
        }

        if (transform != node.get_global_transform())
            node.set_global_transform(transform);

        if (keyboard->was_pressed(Keyboard::Key::Space)) {
            float new_time_scale = engine.get_time().is_paused() ? 1.0f : 0.0f;
            engine.get_time().set_time_scale(new_time_scale);
        }
    }

    static inline void navigate_callback(Engine& engine, void* state) {
        static_cast<Navigation*>(state)->navigate(engine);
    }

private:
    SceneNodes::UID m_node_ID;
    float m_vertical_rotation;
    float m_horizontal_rotation;
    float m_velocity;
};

class CameraHandler final {
public:
    CameraHandler(Cameras::UID camera_ID, float aspect_ratio)
        : m_camera_ID(camera_ID), m_aspect_ratio(aspect_ratio), m_FOV(PI<float>() / 4.0f) {
        Matrix4x4f perspective_matrix, inverse_perspective_matrix;
        CameraUtils::compute_perspective_projection(0.1f, 100.0f, m_FOV, m_aspect_ratio,
            perspective_matrix, inverse_perspective_matrix);

        Cameras::set_projection_matrices(m_camera_ID, perspective_matrix, inverse_perspective_matrix);
    }

    void handle(const Engine& engine) {

        const Mouse* mouse = engine.get_mouse();
        float new_FOV = m_FOV - mouse->get_scroll_delta() * engine.get_time().get_smooth_delta_time(); // TODO Non-linear increased / decrease. Especially that it can become negative is an issue.

        float window_aspect_ratio = engine.get_window().get_aspect_ratio();
        if (window_aspect_ratio != m_aspect_ratio || new_FOV != m_FOV) {
            Matrix4x4f perspective_matrix, inverse_perspective_matrix;
            CameraUtils::compute_perspective_projection(0.1f, 100.0f, m_FOV, m_aspect_ratio,
                perspective_matrix, inverse_perspective_matrix);

            Cameras::set_projection_matrices(m_camera_ID, perspective_matrix, inverse_perspective_matrix);
            m_aspect_ratio = window_aspect_ratio;
            m_FOV = new_FOV;
        }
    }

    static inline void handle_callback(Engine& engine, void* state) {
        static_cast<CameraHandler*>(state)->handle(engine);
    }

private:
    Cameras::UID m_camera_ID;
    float m_aspect_ratio;
    float m_FOV;
};

static inline void update_FPS(Engine& engine, void* state) {
    static const int COUNT = 8;
    static float delta_times[COUNT] = { 1e30f, 1e30f, 1e30f, 1e30f, 1e30f, 1e30f, 1e30f, 1e30f };
    static int next_index = 0;

    delta_times[next_index] = engine.get_time().get_raw_delta_time();
    next_index = (next_index + 1) % COUNT;

    float summed_deltas = 0.0f;
    for (int i = 0; i < COUNT; ++i)
        summed_deltas += delta_times[i];
    float fps = COUNT / summed_deltas;

    std::ostringstream title;
    title << "SimpleViewer - FPS " << fps;
    engine.get_window().set_name(title.str().c_str());
}

// Merges all nodes in the scene sharing the same material and destroys all other nodes.
// Future work
// * Only combine meshes within some max distance of each other, fx the diameter of their bounds.
//   This avoids their bounding boxes containing mostly empty space and messing up ray tracing, 
//   which would be the case if two models on opposite sides of the scene where to be combined.
//   Profile if it makes a difference or if OptiX doesn't care.
void mesh_combine_whole_scene(SceneNodes::UID scene_root) {

    // Asserts of properties used when combining UIDs and mesh flags in one uint key.
    assert(MeshModels::UID::MAX_IDS <= 0xFFFFFF);
    assert(MeshFlags::Position <= 0xFF);
    assert(MeshFlags::Normal <= 0xFF);
    assert(MeshFlags::Texcoord <= 0xFF);
    
    std::vector<bool> used_meshes = std::vector<bool>(Meshes::capacity());
    for (Meshes::UID mesh_ID : Meshes::get_iterable())
        used_meshes[mesh_ID] = false;

    struct OrderedModel {
        unsigned int key;
        MeshModels::UID model_ID;

        inline bool operator<(OrderedModel lhs) const { return key < lhs.key; }
    };

    // Sort models based on material ID and mesh flags.
    std::vector<OrderedModel> ordered_models = std::vector<OrderedModel>();
    ordered_models.reserve(MeshModels::capacity());
    for (MeshModels::UID model_ID : MeshModels::get_iterable()) {
        unsigned int key = MeshModels::get_material_ID(model_ID).get_index() << 8u;

        // Least significant bits in key consist of mesh flags.
        Mesh mesh = MeshModels::get_mesh_ID(model_ID);
        key |= mesh.get_positions() ? MeshFlags::Position : MeshFlags::None;
        key |= mesh.get_normals() ? MeshFlags::Normal : MeshFlags::None;
        key |= mesh.get_texcoords() ? MeshFlags::Texcoord : MeshFlags::None;

        OrderedModel model = { key, model_ID };
        ordered_models.push_back(model);
    }

    std::sort(ordered_models.begin(), ordered_models.end());

    { // Loop through models, merging all models in a segment with same material and flags.
        auto segment_begin = ordered_models.begin();
        for (auto itr = ordered_models.begin(); itr < ordered_models.end(); ++itr) {
            bool next_material_found = itr->key != segment_begin->key;
            bool last_model = (itr + 1) == ordered_models.end();
            if (next_material_found || last_model) {
                auto segment_end = itr;
                if (last_model)
                    ++segment_end;
                // Combine the meshes in the segment if there are more than one.
                auto model_count = segment_end - segment_begin;
                if (model_count == 1) {
                    Meshes::UID mesh_ID = MeshModels::get_mesh_ID(segment_begin->model_ID);
                    used_meshes[mesh_ID] = true;
                }

                if (model_count > 1) {
                    Material material = MeshModels::get_material_ID(segment_begin->model_ID);

                    // Create new scene node to hold the combined model.
                    SceneNode node0 = MeshModels::get_scene_node_ID(segment_begin->model_ID);
                    Transform transform = node0.get_global_transform();
                    SceneNode node = SceneNodes::create(material.get_name() + "_combined", transform);
                    node.set_parent(scene_root);

                    std::vector<MeshUtils::TransformedMesh> transformed_meshes = std::vector<MeshUtils::TransformedMesh>();
                    for (auto model = segment_begin; model < segment_end; ++model) {
                        Meshes::UID mesh_ID = MeshModels::get_mesh_ID(model->model_ID);
                        SceneNode node = MeshModels::get_scene_node_ID(model->model_ID);
                        MeshUtils::TransformedMesh meshie = { mesh_ID, node.get_global_transform() };
                        transformed_meshes.push_back(meshie);
                    }

                    std::string mesh_name = material.get_name() + "_combined_mesh";
                    unsigned int mesh_flags = segment_begin->key; // The mesh flags are contained in the key.
                    Meshes::UID merged_mesh_ID = MeshUtils::combine(mesh_name, transformed_meshes.data(), transformed_meshes.data() + transformed_meshes.size(), mesh_flags);

                    // Create new model.
                    MeshModels::UID merged_model = MeshModels::create(node.get_ID(), merged_mesh_ID, material.get_ID());
                    if (merged_mesh_ID.get_index() < used_meshes.size())
                        used_meshes[merged_model] = true;
                }

                segment_begin = itr;
            }
        }
    }

    // Destroy meshes that are no longer used.
    // NOTE Reference counting on the mesh UIDs would be really handy here.
    for (Meshes::UID mesh_ID : Meshes::get_iterable())
        if (mesh_ID.get_index() < used_meshes.size() && used_meshes[mesh_ID] == false)
            Meshes::destroy(mesh_ID);

    // Destroy old models and scene nodes that no longer connect to a mesh.
    // TODO Delete parents as well.
    for (OrderedModel ordered_model : ordered_models) {
        MeshModel model = ordered_model.model_ID;
        if (!model.get_mesh().exists()) {
            SceneNodes::destroy(model.get_scene_node().get_ID());
            MeshModels::destroy(model.get_ID());
        }
    }
}

static inline void scenenode_cleanup_callback(void* dummy) {
    Images::reset_change_notifications();
    LightSources::reset_change_notifications();
    Materials::reset_change_notifications();
    Meshes::reset_change_notifications();
    MeshModels::reset_change_notifications();
    SceneNodes::reset_change_notifications();
    // Scenes::reset_change_notifications();
    Textures::reset_change_notifications();
}

void initializer(Cogwheel::Core::Engine& engine) {
    engine.get_window().set_name("SimpleViewer");

    Images::allocate(8u);
    LightSources::allocate(8u);
    Materials::allocate(8u);
    Meshes::allocate(8u);
    MeshModels::allocate(8u);
    SceneNodes::allocate(8u);
    Textures::allocate(8u);

    engine.add_tick_cleanup_callback(scenenode_cleanup_callback, nullptr);

    // Setup scene.
    SceneRoots::allocate(1u);
    SceneNodes::UID root_node_ID = SceneNodes::create("Root");
    SceneRoots::UID scene_ID = SceneRoots::UID::invalid_UID();
    if (!g_environment.empty()) {
        Image image = StbImageLoader::load(g_environment);
        if (channel_count(image.get_pixel_format()) != 4) {
            Image new_image = ImageUtils::change_format(image.get_ID(), PixelFormat::RGBA_Float);
            Images::destroy(image.get_ID());
            image = new_image;
        }
        Textures::UID env_ID = Textures::create2D(image.get_ID(), MagnificationFilter::Linear, MinificationFilter::Linear, WrapMode::Repeat, WrapMode::Clamp);
        scene_ID = SceneRoots::create("Model scene", root_node_ID, env_ID);
    } else
        scene_ID = SceneRoots::create("Model scene", root_node_ID, g_environment_color);
    
    // Create camera
    SceneNodes::UID cam_node_ID = SceneNodes::create("Cam");
    Cameras::allocate(1u);
    Cameras::UID cam_ID = Cameras::create(cam_node_ID, scene_ID, Matrix4x4f::identity(), Matrix4x4f::identity()); // Matrices will be set up by the CameraHandler.
    CameraHandler* camera_handler = new CameraHandler(cam_ID, engine.get_window().get_aspect_ratio());
    engine.add_mutating_callback(CameraHandler::handle_callback, camera_handler);

    // Load model
    bool load_model_from_file = false;
    if (g_scene.empty() || g_scene.compare("CornellBox") == 0)
        create_cornell_box_scene(cam_ID, root_node_ID);
    else if (g_scene.compare("MaterialScene") == 0)
        create_material_scene(cam_ID, root_node_ID);
    else if (g_scene.compare("SphereScene") == 0)
        create_sphere_scene(cam_ID, root_node_ID);
    else if (g_scene.compare("TestScene") == 0)
        create_test_scene(engine, cam_ID, root_node_ID);
    else {
        SceneNodes::UID obj_root_ID = ObjLoader::load(g_scene, StbImageLoader::load);
        SceneNodes::set_parent(obj_root_ID, root_node_ID);
        mesh_combine_whole_scene(root_node_ID);
        load_model_from_file = true;
    }

    // Rough approximation of the scene bounds using bounding spheres.
    AABB scene_bounds = AABB::invalid();
    for (MeshModel model : MeshModels::get_iterable()) {
        AABB mesh_aabb = model.get_mesh().get_bounds();
        Transform transform = model.get_scene_node().get_global_transform();
        Vector3f bounding_sphere_center = transform * mesh_aabb.center();
        float bounding_sphere_radius = magnitude(mesh_aabb.size()) * 0.5f;
        AABB global_mesh_aabb = AABB(bounding_sphere_center - bounding_sphere_radius, bounding_sphere_center + bounding_sphere_radius);
        scene_bounds.grow_to_contain(global_mesh_aabb);
    }
    g_scene_size = magnitude(scene_bounds.size());

    float camera_velocity = g_scene_size * 0.1f;
    Navigation* camera_navigation = new Navigation(cam_node_ID, camera_velocity);
    engine.add_mutating_callback(Navigation::navigate_callback, camera_navigation);
    engine.add_mutating_callback(update_FPS, nullptr);

    if (load_model_from_file) {
        Transform cam_transform = SceneNodes::get_global_transform(cam_node_ID);
        cam_transform.translation = scene_bounds.center() + scene_bounds.size();
        cam_transform.look_at(scene_bounds.center());
        SceneNodes::set_global_transform(cam_node_ID, cam_transform);
    }

    // Add a light source if none were added yet.
    bool no_light_sources = LightSources::begin() == LightSources::end() && g_environment.empty();
    if (no_light_sources && load_model_from_file) {
        Quaternionf light_direction = Quaternionf::look_in(normalize(Vector3f(-0.1f, -10.0f, -0.1f)));
        Transform light_transform = Transform(Vector3f::zero(), light_direction);
        SceneNodes::UID light_node_ID = SceneNodes::create("Light", light_transform);
        LightSources::UID light_ID = LightSources::create_directional_light(light_node_ID, RGB(15.0f));
        SceneNodes::set_parent(light_node_ID, root_node_ID);
    }
}

void initialize_window(Cogwheel::Core::Window& window) {
    OptiXRenderer::Renderer* renderer = new OptiXRenderer::Renderer();
    // renderer->set_scene_epsilon(g_scene_size * 0.00001f);
    Engine::get_instance()->add_non_mutating_callback(OptiXRenderer::render_callback, renderer);
}

void print_usage() {
    char* usage =
        "usage simpleviewer:\n"
        "  -h | --help: Show command line usage for simpleviewer.\n"
        "  -s | --scene <model>: Loads the model specified. Reserved names are 'CornellBox', 'MaterialScene', 'SphereScene' and 'TestScene', which loads the corresponding builtin scenes.\n"
        "  -e | --environment-map <image>: Loads the specified image for the environment.\n"
        "  -c | --environment-color <RGB>: Sets the background color to the specified value.\n";
    printf("%s", usage);
}

// String representation is assumed to be "[r, g, b]".
RGB parse_RGB(const std::string& rgb_str) {
    const char* red_begin = rgb_str.c_str() + 1; // Skip [
    char* channel_end;
    RGB result = RGB::black();

    result.r = strtof(red_begin, &channel_end);

    char* g_begin = channel_end + 1; // Skip ,
    result.g = strtof(g_begin, &channel_end);

    char* b_begin = channel_end + 1; // Skip ,
    result.b = strtof(b_begin, &channel_end);

    return result;
}

void main(int argc, char** argv) {

    std::string command = g_scene = argc >= 2 ? std::string(argv[1]) : "";
    if (command.compare("-h") == 0 || command.compare("--help") == 0) {
        print_usage();
        return;
    }  

    // Parse command line arguments.
    int argument = 1;
    while (argument < argc) {
        if (strcmp(argv[argument], "--scene") == 0 || strcmp(argv[argument], "-s") == 0)
            g_scene = std::string(argv[++argument]);
        else if (strcmp(argv[argument], "--environment-map") == 0 || strcmp(argv[argument], "-e") == 0)
            g_environment = std::string(argv[++argument]);
        else if (strcmp(argv[argument], "--environment-color") == 0 || strcmp(argv[argument], "-c") == 0)
            g_environment_color = parse_RGB(std::string(argv[++argument]));
        else
            printf("Unknown argument: '%s'\n", argv[argument]);
        ++argument;
    }

    if (g_scene.empty())
        printf("SimpleViewer will display the Cornell Box scene.\n");

    GLFWDriver::run(initializer, initialize_window);
}
