// SimpleViewer.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <CameraHandlers.h>

#include <Scenes/CornellBox.h>
#include <Scenes/Glass.h>
#include <Scenes/Material.h>
#include <Scenes/Opacity.h>
#include <Scenes/Sphere.h>
#include <Scenes/SphereLight.h>
#include <Scenes/Test.h>
#include <Scenes/Veach.h>

#include <GUI/RenderingGUI.h>

#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Core/Engine.h>
#include <Bifrost/Input/Keyboard.h>
#include <Bifrost/Input/Mouse.h>
#include <Bifrost/Scene/Camera.h>
#include <Bifrost/Scene/LightSource.h>
#include <Bifrost/Scene/SceneNode.h>
#include <Bifrost/Scene/SceneRoot.h>

#include <ImGui/ImGUIAdaptor.h>
#include <ImGui/Renderers/DX11Renderer.h>

#include <Win32Driver.h>
#include <DX11Renderer/Compositor.h>
#include <DX11Renderer/Renderer.h>
#ifdef OPTIX_FOUND
#include <DX11OptiXAdaptor/Adaptor.h>
#include <OptiXRenderer/Renderer.h>
#endif

#include <glTFLoader/glTFLoader.h>
#include <ObjLoader/ObjLoader.h>
#include <StbImageLoader/StbImageLoader.h>

#include <iostream>
#include <io.h>

using namespace Bifrost::Assets;
using namespace Bifrost::Core;
using namespace Bifrost::Input;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;

static std::string g_scene_name;
static std::string g_environment;
static RGB g_environment_color = RGB(0.68f, 0.92f, 1.0f);
static DX11Renderer::Compositor* compositor = nullptr;
static DX11Renderer::Renderer* dx11_renderer = nullptr;
static ImGui::ImGuiAdaptor* imgui = nullptr;
static CameraNavigation* camera_navigation = nullptr;
static CameraViewportHandler* camera_handler;
static Vector2ui g_window_size = Vector2ui(640, 480);

static Vector3f g_camera_translation = Vector3f(nanf(""));
static float g_camera_horizontal_rotation = nanf("");
static float g_camera_vertical_rotation = nanf("");

OptiXRenderer::Renderer* optix_renderer = nullptr;

static inline void update_FPS(Engine& engine) {
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

Image load_image(const std::string& path) {
    const int read_only_flag = 4;
    if (_access(path.c_str(), read_only_flag) >= 0)
        return StbImageLoader::load(path);
    std::string new_path = path;

    // Test tga.
    new_path[path.size() - 3] = 't'; new_path[path.size() - 2] = 'g'; new_path[path.size() - 1] = 'a';
    if (_access(new_path.c_str(), read_only_flag) >= 0)
        return StbImageLoader::load(new_path);

    // Test png.
    new_path[path.size() - 3] = 'p'; new_path[path.size() - 2] = 'n'; new_path[path.size() - 1] = 'g';
    if (_access(new_path.c_str(), read_only_flag) >= 0)
        return StbImageLoader::load(new_path);

    // Test jpg.
    new_path[path.size() - 3] = 'j'; new_path[path.size() - 2] = 'p'; new_path[path.size() - 1] = 'g';
    if (_access(new_path.c_str(), read_only_flag) >= 0)
        return StbImageLoader::load(new_path);

    // No dice. Report error and return an invalid ID.
    printf("No image found at '%s'\n", path.c_str());
    return Image::invalid();
}

// Merges all nodes in the scene sharing the same material and destroys all other nodes.
// Future work
// * Only combine meshes within some max distance to each other, fx the diameter of their bounds.
//   This avoids their bounding boxes containing mostly empty space and messing up ray tracing, 
//   which would be the case if two models on opposite sides of the scene were to be combined.
//   It also avoids combining leafs on a tree acros the entire scene.
void mesh_combine_whole_scene(SceneNodeID scene_root) {

    // Asserts of properties used when combining UIDs and mesh flags in one uint key.
    assert(MeshModelID::MAX_IDS <= 0xFFFFFF);
    assert((int)MeshFlag::Position <= 0xFF);
    assert((int)MeshFlag::Normal <= 0xFF);
    assert((int)MeshFlag::Texcoord <= 0xFF);
    
    std::vector<bool> used_meshes = std::vector<bool>(Meshes::capacity());
    for (MeshID mesh_ID : Meshes::get_iterable())
        used_meshes[mesh_ID] = false;

    struct OrderedModel {
        unsigned int key;
        MeshModelID model_ID;

        inline bool operator<(OrderedModel lhs) const { return key < lhs.key; }
    };

    // Sort models based on material ID and mesh flags.
    std::vector<OrderedModel> ordered_models = std::vector<OrderedModel>();
    ordered_models.reserve(MeshModels::capacity());
    for (MeshModelID model_ID : MeshModels::get_iterable()) {
        unsigned int key = MeshModels::get_material_ID(model_ID).get_index() << 8u;

        // Least significant bits in key consist of mesh flags.
        Mesh mesh = MeshModels::get_mesh_ID(model_ID);
        key |= mesh.get_flags().raw();

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
                    MeshID mesh_ID = MeshModels::get_mesh_ID(segment_begin->model_ID);
                    used_meshes[mesh_ID] = true;
                }

                if (model_count > 1) {
                    Material material = MeshModels::get_material_ID(segment_begin->model_ID);

                    // Create new scene node to hold the combined model.
                    SceneNode node0 = MeshModels::get_scene_node_ID(segment_begin->model_ID);
                    Transform transform = node0.get_global_transform();
                    SceneNode merged_node = SceneNode(material.get_name() + "_combined", transform);
                    merged_node.set_parent(scene_root);

                    std::vector<MeshUtils::TransformedMesh> transformed_meshes = std::vector<MeshUtils::TransformedMesh>();
                    for (auto model = segment_begin; model < segment_end; ++model) {
                        MeshID mesh_ID = MeshModels::get_mesh_ID(model->model_ID);
                        SceneNode node = MeshModels::get_scene_node_ID(model->model_ID);
                        MeshUtils::TransformedMesh meshie = { mesh_ID, node.get_global_transform() };
                        transformed_meshes.push_back(meshie);
                    }

                    std::string mesh_name = material.get_name() + "_combined_mesh";
                    unsigned int mesh_flags = segment_begin->key; // The mesh flags are contained in the key.
                    Mesh merged_mesh = MeshUtils::combine(mesh_name, transformed_meshes.data(), transformed_meshes.data() + transformed_meshes.size(), mesh_flags);

                    // Create new model.
                    MeshModel(merged_node, merged_mesh, material);
                    if (merged_mesh.get_ID().get_index() < used_meshes.size())
                        used_meshes[merged_mesh.get_ID()] = true;
                }

                segment_begin = itr;
            }
        }
    }

    // Destroy meshes that are no longer used.
    // NOTE Reference counting on the mesh UIDs would be really handy here.
    for (MeshID mesh_ID : Meshes::get_iterable())
        if (mesh_ID.get_index() < used_meshes.size() && used_meshes[mesh_ID] == false)
            Meshes::destroy(mesh_ID);

    // Destroy old models and scene nodes that no longer connect to a mesh.
    // TODO Delete parents as well.
    for (OrderedModel ordered_model : ordered_models) {
        MeshModel model = ordered_model.model_ID;
        if (!model.get_mesh().exists()) {
            model.get_scene_node().destroy();
            model.destroy();
        }
    }
}

void detect_and_flag_cutout_materials() {
    // A cutout is a black / white alpha mask. In order to allow for textures with 'soft edges' to be flagged as cut outs 
    // (because transparency is a pain) we allow soft borders.
    // These are detected by grouping pixels in 2x2 groups. If a single pixel in that group is non-grey, then the group is considered a cutout.

    enum State : unsigned char { Unprocessed, Cutout, Transparent};

    std::vector<State> image_states = std::vector<State>();
    image_states.resize(Images::capacity());
    memset(image_states.data(), Unprocessed, image_states.capacity());

    for (MeshModel model : MeshModels::get_iterable()) {
        Material material = model.get_material();
        if (material.get_coverage_texture().exists()) {
            Image coverage_img = material.get_coverage_texture().get_image();
            assert(coverage_img.get_pixel_format() == PixelFormat::Alpha8);

            State& image_state = image_states[coverage_img.get_ID()];
            if (image_state == Unprocessed)
            {
                int width = coverage_img.get_width(), height = coverage_img.get_height();
                unsigned char* pixels = coverage_img.get_pixels<unsigned char>();

                auto is_cutout_opacity = [](unsigned char intensity) -> bool { return intensity < 2 || 253 < intensity; };

                image_state = Cutout;
                for (int y = 0; y < height - 1; ++y)
                    for (int x = 0; x < width - 1; ++x) {
                        unsigned char intensity = pixels[x + y * width];
                        if (!is_cutout_opacity(intensity)) {
                            // Intensity is not black / white.
                            // Check if the pixel is part of a border or if its part of a larger 'greyish blob'.

                            bool cutout_border = is_cutout_opacity(pixels[(x + 1) + y * width])
                                || is_cutout_opacity(pixels[x + (y + 1) * width])
                                || is_cutout_opacity(pixels[(x + 1) + (y + 1) * width]);
                            if (!cutout_border)
                                image_state = Transparent;
                        }
                    }
            }

            if (image_state == Cutout)
                material.set_flags(MaterialFlag::Cutout);
        }
    }
}

void main_loop(Engine& engine) {
    const Keyboard* keyboard = engine.get_keyboard();

    // Update window title with FPS
    update_FPS(engine);

    { // ImGUI callback
        bool control_pressed = keyboard->is_pressed(Keyboard::Key::LeftControl) || keyboard->is_pressed(Keyboard::Key::RightControl);
        bool g_was_released = keyboard->was_released(Keyboard::Key::G);
        imgui->set_enabled(imgui->is_enabled() ^ (g_was_released && control_pressed));

        imgui->new_frame(engine);
    }

    // Toggle the renderer of the main camera
    if (keyboard->was_released(Keyboard::Key::P) && !keyboard->is_modifiers_pressed()) {
        CameraID first_camera_ID = *Cameras::begin();
        Renderers::Iterator renderer_itr = Renderers::get_iterator(Cameras::get_renderer_ID(first_camera_ID));
        ++renderer_itr;
        Renderers::Iterator new_renderer_itr = (renderer_itr == Renderers::end()) ? Renderers::begin() : renderer_itr;
        Cameras::set_renderer_ID(first_camera_ID, *new_renderer_itr);
    }

    // Handle camera navigation and frustum.
    camera_navigation->navigate(engine);
    camera_handler->handle(engine);
}

static inline void miniheaps_cleanup_callback() {
    Cameras::reset_change_notifications();
    Images::reset_change_notifications();
    LightSources::reset_change_notifications();
    Materials::reset_change_notifications();
    Meshes::reset_change_notifications();
    MeshModels::reset_change_notifications();
    SceneNodes::reset_change_notifications();
    SceneRoots::reset_change_notifications();
    Textures::reset_change_notifications();
}

int initializer(Engine& engine) {
    engine.get_window().set_name("SimpleViewer");

    Cameras::allocate(1u);
    Images::allocate(8u);
    LightSources::allocate(8u);
    Materials::allocate(8u);
    Meshes::allocate(8u);
    MeshModels::allocate(8u);
    Renderers::allocate(2u);
    SceneNodes::allocate(8u);
    SceneRoots::allocate(1u);
    Textures::allocate(8u);

    engine.add_tick_cleanup_callback(miniheaps_cleanup_callback);

    return 0;
}

int initialize_scene(Engine& engine, ImGui::ImGuiAdaptor* imgui) {
    // Setup scene.
    SceneRoot scene = SceneRoot::invalid();
    if (!g_environment.empty()) {
        Image image = StbImageLoader::load(g_environment);
        if (image.exists()) {
            if (channel_count(image.get_pixel_format()) != 4) {
                // OptiXRenderer requires four channel environment maps
                PixelFormat format = image.is_sRGB() ? PixelFormat::RGBA32 : PixelFormat::RGBA_Float;
                image.change_format(format, image.is_sRGB());
            }
            Texture env = Texture::create2D(image, MagnificationFilter::Linear, MinificationFilter::Linear, WrapMode::Repeat, WrapMode::Clamp);
            scene = SceneRoot("Model scene", env);
        } else
            scene = SceneRoot("Model scene", g_environment_color);
    } else
        scene = SceneRoot("Model scene", g_environment_color);
    SceneNode root_node = scene.get_root_node();

    // Create camera. Matrices will be set up by the CameraViewportHandler.
    CameraID cam_ID = Cameras::create("Camera", scene.get_ID(), Matrix4x4f::identity(), Matrix4x4f::identity());
    camera_handler = new CameraViewportHandler(cam_ID, engine.get_window().get_aspect_ratio(), 0.1f, 100.0f);
#ifdef OPTIX_FOUND
    // High bounce count to support test scenes with lots of glass or many total internal reflection bounces.
    optix_renderer->set_max_bounce_count(cam_ID, 32);
#endif

    // Load model
    auto resource_directory = engine.data_directory() / "SimpleViewer" / "Resources";
    bool load_model_from_file = false;
    if (g_scene_name.empty() || g_scene_name.compare("CornellBox") == 0)
        Scenes::create_cornell_box(cam_ID, root_node);
    else if (g_scene_name.compare("GlassScene") == 0)
        Scenes::create_glass_scene(cam_ID, root_node, resource_directory);
    else if (g_scene_name.compare("MaterialScene") == 0)
        Scenes::create_material_scene(cam_ID, root_node, imgui, resource_directory);
    else if (g_scene_name.compare("OpacityScene") == 0)
        Scenes::create_opacity_scene(engine, cam_ID, root_node);
    else if (g_scene_name.compare("SphereScene") == 0)
        Scenes::create_sphere_scene(cam_ID, root_node);
    else if (g_scene_name.compare("SphereLightScene") == 0)
        Scenes::SphereLightScene::create(engine, cam_ID, scene);
    else if (g_scene_name.compare("TestScene") == 0)
        Scenes::create_test_scene(engine, cam_ID, root_node, resource_directory);
    else if (g_scene_name.compare("VeachScene") == 0)
        Scenes::create_veach_scene(engine, cam_ID, scene);
    else {
#ifdef OPTIX_FOUND
        // Conservative bounce count, as the loaded scenes could be very heavy on bounces and geometry.
        optix_renderer->set_max_bounce_count(cam_ID, 4);
#endif

        printf("Loading scene: '%s'\n", g_scene_name.c_str());
        SceneNode obj_root = SceneNode::invalid();
        if (ObjLoader::file_supported(g_scene_name))
            obj_root = ObjLoader::load(g_scene_name, load_image);
        else if (glTFLoader::file_supported(g_scene_name))
            obj_root = glTFLoader::load(g_scene_name);
        obj_root.set_parent(root_node);
        // mesh_combine_whole_scene(root_node_ID);
        detect_and_flag_cutout_materials();
        load_model_from_file = true;
    }

    if (root_node.get_children().size() == 0u) {
        printf("SimpleViewer error: No objects in scene.\n");
        return -1;
    }

    // Rough approximation of the scene bounds using bounding spheres.
    AABB scene_bounds = AABB::invalid();
    for (MeshModel model : MeshModels::get_iterable()) {
        AABB mesh_aabb = model.get_mesh().get_bounds();
        Transform transform = model.get_scene_node().get_global_transform();
        Vector3f bounding_sphere_center = transform * mesh_aabb.center();
        float bounding_sphere_radius = magnitude(mesh_aabb.size() * transform.scale) * 0.5f;
        AABB global_mesh_aabb = AABB(bounding_sphere_center - bounding_sphere_radius, bounding_sphere_center + bounding_sphere_radius);
        scene_bounds.grow_to_contain(global_mesh_aabb);
    }

    if (load_model_from_file) {
        Transform cam_transform = Cameras::get_transform(cam_ID);
        cam_transform.translation = scene_bounds.center() + scene_bounds.size();
        cam_transform.look_at(scene_bounds.center());
        Cameras::set_transform(cam_ID, cam_transform);
    }

    // Add a light source if none were added yet.
    bool no_light_sources = LightSources::get_iterable().is_empty() && g_environment.empty();
    if (no_light_sources && load_model_from_file) {
        Quaternionf light_direction = Quaternionf::look_in(normalize(Vector3f(-0.1f, -10.0f, -0.1f)));
        Transform light_transform = Transform(Vector3f::zero(), light_direction);
        SceneNode light_node = SceneNode("Light", light_transform);
        DirectionalLight(light_node, RGB(15.0f));
        light_node.set_parent(root_node);
    }

    float scene_size = magnitude(scene_bounds.size());
    camera_handler->set_near_and_far(scene_size / 10000.0f, scene_size * 3.0f);

    float camera_velocity = scene_size * 0.1f;
    camera_navigation = new CameraNavigation(cam_ID, camera_velocity, g_camera_translation, g_camera_vertical_rotation, g_camera_horizontal_rotation);

    return 0;
}

int win32_window_initialized(Engine& engine, Window& window, HWND& hwnd) {
    using namespace DX11Renderer;

    if (window.get_width() != g_window_size.x || window.get_height() != g_window_size.y)
        window.resize(g_window_size.x, g_window_size.y);

    compositor = Compositor::initialize(hwnd, window);

    dx11_renderer = (Renderer*)compositor->add_renderer(Renderer::initialize).get();

#ifdef OPTIX_FOUND
    auto* optix_adaptor = (DX11OptiXAdaptor::Adaptor*)compositor->add_renderer(DX11OptiXAdaptor::Adaptor::initialize).get();
    optix_renderer = optix_adaptor->get_renderer();
#endif

    engine.add_non_mutating_callback([=] { compositor->render(); });

    imgui = new ImGui::ImGuiAdaptor();

    int scene_result = initialize_scene(engine, imgui);
    if (scene_result < 0)
        return scene_result;

    { // Setup GUI
        imgui->add_frame(std::make_unique<GUI::RenderingGUI>(camera_navigation, compositor, dx11_renderer, optix_renderer));
        compositor->add_GUI_renderer(ImGui::Renderers::DX11Renderer::initialize);
    }

    engine.add_mutating_callback([&engine] { main_loop(engine); });

    return  0;
}

void print_usage() {
    char* usage =
        "usage simpleviewer:\n"
        "  -h  | --help: Show command line usage for simpleviewer.\n"
        "  -s  | --scene <model>: Loads the model specified. Reserved names are 'CornellBox', 'GlassScene', 'MaterialScene', 'SphereScene', 'SphereLightScene', 'TestScene' and 'VeachScene', which loads the corresponding builtin scenes.\n"
        "  -e  | --environment-map <image>: Loads the specified image for the environment.\n"
        "  -c  | --environment-tint [R,G,B]: Tint the environment by the specified value.\n"
        "      | --window-size [width, height]: Size of the window.\n"
        "      | --camera-position [X,Y,Z]: Position of the camera.\n"
        "      | --camera-rotation [vertical, horizontal]: Orientation of the camera in radians.\n";
    printf("%s", usage);
}

// String representation is assumed to be "[x, y]".
inline Vector2f parse_vector2f(const char* const vec2_str) {
    Vector2f result;

    const char* x_begin = vec2_str + 1; // Skip [
    char* channel_end;
    result.x = strtof(x_begin, &channel_end);

    char* y_begin = channel_end + 1; // Skip ,
    result.y = strtof(y_begin, &channel_end);

    return result;
}

// String representation is assumed to be "[x, y, z]".
inline Vector3f parse_vector3f(const char* const vec3_str) {
    Vector3f result;

    const char* x_begin = vec3_str + 1; // Skip [
    char* channel_end;
    result.x = strtof(x_begin, &channel_end);

    char* y_begin = channel_end + 1; // Skip ,
    result.y = strtof(y_begin, &channel_end);

    char* z_begin = channel_end + 1; // Skip ,
    result.z = strtof(z_begin, &channel_end);

    return result;
}

// String representation is assumed to be "[r, g, b]".
inline RGB parse_RGB(const char* const rgb_str) {
    Vector3f vec = parse_vector3f(rgb_str);
    return RGB(vec.x, vec.y, vec.z);
}

int main(int argc, char** argv) {

    std::string command = argc >= 2 ? std::string(argv[1]) : "";
    if (command.compare("-h") == 0 || command.compare("--help") == 0) {
        print_usage();
        return 0;
    }

    // Parse command line arguments.
    int argument = 1;
    while (argument < argc) {
        if (strcmp(argv[argument], "--scene") == 0 || strcmp(argv[argument], "-s") == 0)
            g_scene_name = std::string(argv[++argument]);
        else if (strcmp(argv[argument], "--environment-map") == 0 || strcmp(argv[argument], "-e") == 0)
            g_environment = std::string(argv[++argument]);
        else if (strcmp(argv[argument], "--environment-tint") == 0 || strcmp(argv[argument], "-c") == 0)
            g_environment_color = parse_RGB(argv[++argument]);
        else if (strcmp(argv[argument], "--window-size") == 0)
            g_window_size = (Vector2ui)parse_vector2f(argv[++argument]);
        else if (strcmp(argv[argument], "--camera-position") == 0)
            g_camera_translation = parse_vector3f(argv[++argument]);
        else if (strcmp(argv[argument], "--camera-rotation") == 0) {
            Vector2f camera_rotation = parse_vector2f(argv[++argument]);
            g_camera_vertical_rotation = camera_rotation.x;
            g_camera_horizontal_rotation = camera_rotation.y;
        }
        else
            printf("Unknown argument: '%s'\n", argv[argument]);
        ++argument;
    }

    if (g_scene_name.empty())
        printf("SimpleViewer will display the Cornell Box scene.\n");

    int error_code = Win32Driver::run(initializer, win32_window_initialized);

    delete compositor;

    return error_code;
}
