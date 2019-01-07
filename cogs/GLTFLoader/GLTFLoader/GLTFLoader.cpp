// Cogwheel gltf model loader.
// ------------------------------------------------------------------------------------------------
// Copyright (C), Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#pragma warning(disable : 4267) // Implicit conversion from size_t to int in tiny_gltf.h

#include <GLTFLoader/GLTFLoader.h>

#include <Cogwheel/Assets/Material.h>
#include <Cogwheel/Assets/Mesh.h>
#include <Cogwheel/Assets/MeshModel.h>
#include <Cogwheel/Math/Conversions.h>

#include <StbImageLoader/StbImageLoader.h>

#define TINYGLTF_IMPLEMENTATION 1
#define TINYGLTF_NO_STB_IMAGE 1
#define TINYGLTF_NO_STB_IMAGE_WRITE 1
#include <GLTFLoader/tiny_gltf.h>

using namespace Cogwheel::Assets;
using namespace Cogwheel::Core;
using namespace Cogwheel::Math;
using namespace Cogwheel::Scene;

using Matrix3x3d = Matrix3x3<double>;

namespace GLTFLoader {

inline bool string_ends_with(const std::string& s, const std::string& end) {
    if (s.length() < end.length())
        return false;

    return s.compare(s.length() - end.length(), end.length(), end) == 0;
}

inline MagnificationFilter convert_magnification_filter(int gltf_magnification_filter) {
    bool is_nearest = gltf_magnification_filter == TINYGLTF_TEXTURE_FILTER_NEAREST;
    return is_nearest ? MagnificationFilter::None : MagnificationFilter::Linear;
}

inline MinificationFilter convert_minification_filter(int gltf_minification_filter) {
    switch (gltf_minification_filter) {
    case TINYGLTF_TEXTURE_FILTER_NEAREST:
        return MinificationFilter::None;
    case TINYGLTF_TEXTURE_FILTER_LINEAR:
        return MinificationFilter::Linear;
    case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST:
        printf("GLTFLoader::load warning: Unsupported minification filter NEAREST_MIPMAP_NEAREST. Using LINEAR_MIPMAP_LINEAR.\n");
        return MinificationFilter::Trilinear;
    case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST:
        printf("GLTFLoader::load warning: Unsupported minification filter LINEAR_MIPMAP_NEAREST. Using LINEAR_MIPMAP_LINEAR.\n");
        return MinificationFilter::Trilinear;
    case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR:
        printf("GLTFLoader::load warning: Unsupported minification filter NEAREST_MIPMAP_LINEAR. Using LINEAR_MIPMAP_LINEAR.\n");
        return MinificationFilter::Trilinear;
    case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR:
        return MinificationFilter::Trilinear;
    default:
        printf("GLTFLoader::load warning: Unknown minification filter mode %u.\n", gltf_minification_filter);
        return MinificationFilter::Trilinear;
    }
}

inline WrapMode convert_wrap_mode(int gltf_wrap_mode) {
    if (gltf_wrap_mode == TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE)
        return WrapMode::Clamp;
    else if (gltf_wrap_mode == TINYGLTF_TEXTURE_WRAP_REPEAT)
        return WrapMode::Repeat;
    else if (gltf_wrap_mode == TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT)
        printf("GLTFLoader::load error: Mirrored repeat wrap mode not supported.\n");
    return WrapMode::Repeat;
}

SceneNodes::UID import_node(const tinygltf::Model& model, const tinygltf::Node& node, const Transform parent_transform,
                            const std::vector<int>& meshes_start_index, const std::vector<Mesh>& meshes,
                            const std::vector<Materials::UID>& material_IDs) {
    // Local transform and scene node.
    Transform local_transform;
    if (node.matrix.size() == 16) {
        // Decompose into scale, rotation and translation.
        const auto& m = node.matrix;
        Matrix3x3d linear_part = { m[0], m[4], m[8],
                                   m[1], m[5], m[9],
                                   m[2], m[6], m[10] };

        Vector3d scales = { magnitude(linear_part.get_column(0)), magnitude(linear_part.get_column(1)), magnitude(linear_part.get_column(2)) };
        float scale = float(scales.x); // TODO Extract min scale and apply non-uniform scaling.

        // Normalize each row.
        for (int r = 0; r < 3; ++r)
            linear_part.set_column(r, normalize(linear_part.get_column(r)));
        Quaterniond rotation = to_quaternion(linear_part);
        Vector3f translation = Vector3f(Vector3d(m[12], m[13], m[14]));
        local_transform = Transform(translation, Quaternionf(rotation), scale);
    } else {
        const auto& s = node.scale;
        float scale = s.size() == 0 ? 1.0f : float((s[0] + s[1] + s[2]) / 3.0); // Only uniform scaling supported.
        const auto& r = node.rotation;
        Quaternionf rotation = r.size() == 0 ? Quaternionf::identity() : Quaterniond(r[0], r[1], r[2], r[3]);
        const auto& t = node.translation;
        Vector3f translation = t.size() == 0 ? Vector3f::zero() : Vector3f(Vector3d(t[0], t[1], t[2]));
        local_transform = Transform(translation, rotation, scale);
    }

    // X-coord is negated as glTF uses a right-handed coordinate system and Cogwheel a right-handed.
    // This also means that we have to invert some of the entries in the quaternion, specifically .x and the rotation direction in .w.
    local_transform.rotation.x = -local_transform.rotation.x;
    local_transform.rotation.w = -local_transform.rotation.w;
    local_transform.translation.x = -local_transform.translation.x;

    Transform global_transform = parent_transform * local_transform;
    SceneNodes::UID scene_node_ID = SceneNodes::create(node.name, global_transform);

    if (node.mesh >= 0) {
        // Get loaded mesh indices.
        int gltf_mesh_index = node.mesh;
        int mesh_index = meshes_start_index[gltf_mesh_index];
        int mesh_end_index = meshes_start_index[gltf_mesh_index + 1];

        for (const auto& primitive : model.meshes[gltf_mesh_index].primitives) {
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
                printf("GLTFLoader::load warning: %s primitive %u not supported.\n", model.meshes[gltf_mesh_index].name.c_str(), primitive.mode);
                continue;
            }

            // Create model.
            Mesh mesh = meshes[mesh_index++];
            auto material_ID = material_IDs[primitive.material];
            MeshModels::create(scene_node_ID, mesh.get_ID(), material_ID);
        }
    }

    // Recurse over children.
    for (const int child_node_index : node.children) {
        const auto& node = model.nodes[child_node_index];
        SceneNode child_node = import_node(model, node, global_transform, meshes_start_index, meshes, material_IDs);
        child_node.set_parent(scene_node_ID);
    }

    return scene_node_ID;
};

template<typename T>
void copy_indices(unsigned int* cogwheel_indices, const T* gltf_indices, size_t count, int index_byte_stride) {
    int gltf_index_T_stride = index_byte_stride / sizeof(T);
    for (size_t i = 0; i < count; ++i) {
        *cogwheel_indices = *gltf_indices;
        ++cogwheel_indices;
        gltf_indices += gltf_index_T_stride;
    }
}

// ------------------------------------------------------------------------------------------------
// Loads a gltf file.
// ------------------------------------------------------------------------------------------------
SceneNodes::UID load(const std::string& filename) {

    // See https://github.com/syoyo/tinygltf/blob/master/loader_example.cc

    tinygltf::Model model;
    tinygltf::TinyGLTF gltf_ctx;
    std::string errors, warnings;

    static auto image_loader = [](tinygltf::Image* gltf_image, std::string* error, std::string* warning, int req_width, int req_height,
                                  const unsigned char *bytes, int size, void* user_data) -> bool {

        Image image = StbImageLoader::load_from_memory("Unnamed", bytes, size);

        if (!image.exists()) {
            if (error)
                *error += "GLTFLoader::load error: Failed to load image from memory.\n";
            return false;
        }

        if (image.get_width() < 1 || image.get_height() < 1) {
            Images::destroy(image.get_ID());
            if (error)
                *error += "GLTFLoader::load error: Failed to load image from memory.\n";
            return false;
        }

        if (req_width > 0 && image.get_width() != req_width) {
            Images::destroy(image.get_ID());
            if (error)
                *error += "GLTFLoader::load error: Image width mismatch.\n";
            return false;
        }

        if (req_height > 0 && image.get_height() != req_height) {
            Images::destroy(image.get_ID());
            if (error)
                *error += "GLTFLoader::load error: Image height mismatch.\n";
            return false;
        }

        // HACK Resize to 4 channels, since OptiXRenderer crashes when there are only three. This should be handled in the OptiXRenderer itself.
        if (channel_count(image.get_pixel_format()) == 3) {
            auto new_image_ID = ImageUtils::change_format(image.get_ID(), PixelFormat::RGBA32);
            Images::destroy(image.get_ID());
            image = new_image_ID;
        }

        gltf_image->width = image.get_width();
        gltf_image->height = image.get_height();
        gltf_image->component = channel_count(image.get_pixel_format());
        // HACK Store image ID in pixels instead of pixel data.
        gltf_image->image.resize(4);
        memcpy(gltf_image->image.data(), &image.get_ID(), sizeof(Images::UID));

        return true;
    };

    gltf_ctx.SetImageLoader(image_loader, nullptr);

    bool ret = false;
    if (string_ends_with(filename, "glb"))
        ret = gltf_ctx.LoadBinaryFromFile(&model, &errors, &warnings, filename.c_str());
    else if (string_ends_with(filename, "gltf"))
        ret = gltf_ctx.LoadASCIIFromFile(&model, &errors, &warnings, filename.c_str());
    else {
        printf("GLTFLoader::load error: '%s' not a glTF file\n", filename.c_str());
        return SceneNodes::UID::invalid_UID();
    }

    if (!warnings.empty())
        printf("GLTFLoader::load warning: %s\n", warnings.c_str());


    if (!errors.empty())
        printf("GLTFLoader::load error: %s\n", errors.c_str());

    if (!ret) {
        printf("GLTFLoader::load error: Failed to parse '%s'\n", filename.c_str());
        return SceneNodes::UID::invalid_UID();
    }

    // Import images.
    auto loaded_image_IDs = std::vector<Images::UID>(model.images.size());
    for (int i = 0; i < model.images.size(); ++i) {
        const auto& gltf_image = model.images[i];
    }

    // Import materials.
    auto loaded_material_IDs = std::vector<Materials::UID>(model.materials.size());
    for (int i = 0; i < model.materials.size(); ++i) {
        const auto& gltf_mat = model.materials[i];
        Materials::Data mat_data = {};
        mat_data.tint = RGB::white();
        mat_data.roughness = 1.0f;
        mat_data.metallic = 0.0f;
        mat_data.coverage = 1.0f;
        for (const auto& val : gltf_mat.values) {
            if (val.first.compare("baseColorFactor") == 0) {
                auto tint = val.second.number_array;
                mat_data.tint = { float(tint[0]), float(tint[1]), float(tint[2]) };
            } else if (val.first.compare("baseColorTexture") == 0) {
                const auto& gltf_texture = model.textures[val.second.TextureIndex()];
                const auto& gltf_image = model.images[gltf_texture.source];
                Images::UID image_ID;
                memcpy(&image_ID, gltf_image.image.data(), sizeof(image_ID));
                assert(Images::has(image_ID));
                Images::set_name(image_ID, gltf_mat.name + "_tint");

                // Create basic texture.
                if (gltf_texture.sampler >= 0) {
                    const auto& gltf_sampler = model.samplers[gltf_texture.sampler];
                    MagnificationFilter magnification_filter = convert_magnification_filter(gltf_sampler.magFilter);
                    MinificationFilter minification_filter = convert_minification_filter(gltf_sampler.minFilter);
                    WrapMode wrapU = convert_wrap_mode(gltf_sampler.wrapR);
                    WrapMode wrapV = convert_wrap_mode(gltf_sampler.wrapS);
                    mat_data.tint_texture_ID = Textures::create2D(image_ID, magnification_filter, minification_filter, wrapU, wrapV);
                } else
                    mat_data.tint_texture_ID = Textures::create2D(image_ID);
            } else if (val.first.compare("roughnessFactor") == 0)
                mat_data.roughness = float(val.second.Factor());
            else if (val.first.compare("metallicFactor") == 0)
                mat_data.metallic = float(val.second.Factor());
        }

        for (const auto& val : gltf_mat.additionalValues) {
            if (val.first.compare("doubleSided") == 0)
                // NOTE to self: Double sided should set a 'thin/doubleSided' property on the meshes instead of on the materials.
                // In case the mesh is used by one sided and two sided meshes, we need to duplicate the mesh.
                printf("GLTFLoader::load warning: doubleSided property not supported.\n");
            else if (val.first.compare("alphaMode") == 0) {
                bool is_cutout = val.second.string_value.compare("MASK") == 0;
                mat_data.flags |= is_cutout ? MaterialFlag::Cutout : MaterialFlag::None;
            }
        }

        loaded_material_IDs[i] = Materials::create(gltf_mat.name, mat_data);
    }

    // Import models.
    auto loaded_meshes_start_index = std::vector<int>();
    loaded_meshes_start_index.reserve(model.meshes.size());
    auto loaded_meshes = std::vector<Mesh>();
    loaded_meshes.reserve(model.meshes.size());
    for (const auto& gltf_mesh : model.meshes) {
        // Mark at what index in loaded_meshes that the models corresponding to the current gltf mesh begins.
        loaded_meshes_start_index.push_back(loaded_meshes.size());
        for (size_t p = 0; p < gltf_mesh.primitives.size(); ++p) {
            const auto& primitive = gltf_mesh.primitives[p];
            
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
                printf("GLTFLoader::load warning: %s[%zu] primitive %u not supported.\n", gltf_mesh.name.c_str(), p, primitive.mode);
                continue;
            }

            // Create mesh.
            unsigned int vertex_count = 0;
            MeshFlags mesh_flags = MeshFlag::None;
            for (auto& attribute : primitive.attributes) {
                const auto& accessor = model.accessors[attribute.second];
                assert(vertex_count == 0 || vertex_count == accessor.count);
                vertex_count = unsigned int(accessor.count);

                if (attribute.first.compare("POSITION") == 0)
                    mesh_flags |= MeshFlag::Position;
                else if (attribute.first.compare("NORMAL") == 0)
                    mesh_flags |= MeshFlag::Normal;
                else if (attribute.first.compare("TEXCOORD_0") == 0)
                    mesh_flags |= MeshFlag::Texcoord;
            }
            assert(mesh_flags.is_set(MeshFlag::Position));

            if (primitive.indices < 0)
                throw new std::exception("GLTFLoader::load error: mesh without primitive indices not supported yet. FIX!\n");
            const tinygltf::Accessor& index_accessor = model.accessors[primitive.indices];
            unsigned int primitive_count = unsigned int(index_accessor.count) / 3;

            Mesh mesh = Meshes::create(gltf_mesh.name, primitive_count, vertex_count, mesh_flags);
            
            { // Import primitve indices.
                assert(index_accessor.type == TINYGLTF_TYPE_SCALAR);
                int element_size = sizeof(unsigned int);
                const auto& buffer_view = model.bufferViews[index_accessor.bufferView];
                const auto& buffer = model.buffers[buffer_view.buffer];

                unsigned int* primitive_indices = (unsigned int*)(void*)mesh.get_primitives();
                
                int buffer_offset = buffer_view.byteOffset + index_accessor.byteOffset;
                int byte_stride = index_accessor.ByteStride(buffer_view);
                assert(byte_stride != -1);

                const unsigned char* src = buffer.data.data() + buffer_offset;
                if (index_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
                    copy_indices(primitive_indices, (unsigned int*)src, index_accessor.count, byte_stride);
                else if (index_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                    copy_indices(primitive_indices, (unsigned short*)src, index_accessor.count, byte_stride);
                else // index_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE
                    copy_indices(primitive_indices, src, index_accessor.count, byte_stride);
            }

            // Import vertex attributes.
            Vector3f min_position = Vector3f();
            Vector3f max_position = Vector3f();
            for (auto& attribute : primitive.attributes) {
                const auto& accessor = model.accessors[attribute.second];
                const auto& buffer_view = model.bufferViews[accessor.bufferView];
                const auto& buffer = model.buffers[buffer_view.buffer];

                void* destination_buffer = nullptr;
                int element_size = -1;
                if (attribute.first.compare("POSITION") == 0) {
                    assert(accessor.type == TINYGLTF_TYPE_VEC3 && accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
                    element_size = sizeof(Vector3f);
                    destination_buffer = mesh.get_positions();
                    auto min_vals = accessor.minValues;
                    min_position = Vector3f(Vector3d(min_vals[0], min_vals[1], min_vals[2]));
                    auto max_vals = accessor.maxValues;
                    max_position = Vector3f(Vector3d(max_vals[0], max_vals[1], max_vals[2]));
                } else if (attribute.first.compare("NORMAL") == 0) {
                    assert(accessor.type == TINYGLTF_TYPE_VEC3 && accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
                    element_size = sizeof(Vector3f);
                    destination_buffer = mesh.get_normals();
                } else if (attribute.first.compare("TEXCOORD_0") == 0) {
                    assert(accessor.type == TINYGLTF_TYPE_VEC2 && accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
                    element_size = sizeof(Vector2f);
                    destination_buffer = mesh.get_texcoords();
                }

                // Only handle known buffers.
                if (destination_buffer != nullptr) {
                    int buffer_offset = buffer_view.byteOffset + accessor.byteOffset;
                    int byte_stride = accessor.ByteStride(buffer_view);
                    assert(byte_stride != -1);

                    const unsigned char* src = buffer.data.data() + buffer_offset;
                    unsigned char* dst = (unsigned char*)destination_buffer;
                    for (unsigned int i = 0; i < vertex_count; ++i) {
                        // Copy element
                        memcpy(dst, src, element_size);
                        dst += element_size;
                        src += byte_stride;
                    }
                }
            }

            { // Negate the mesh's X component as glTF uses a right-handed coordinate system and we use a left-handed.
                // Negate corners of bounding box.
                min_position.x = -min_position.x;
                max_position.x = -max_position.x;

                // Negate positions and normals.
                for (Vector3f& position : mesh.get_position_iterable())
                    position.x = -position.x;
                if (mesh.get_normals() != nullptr)
                    for (Vector3f& normal : mesh.get_normal_iterable())
                        normal.x = -normal.x;

                for (Vector3ui& primitve_index : mesh.get_primitive_iterable())
                    std::swap(primitve_index.x, primitve_index.y);
            }

            mesh.set_bounds(AABB(min_position, max_position));

            loaded_meshes.push_back(mesh);
        }
    }
    // Finally push the total number of meshes to allow fetching begin and end indices as [index] and [index+1]
    loaded_meshes_start_index.push_back(loaded_meshes.size());

    // Import lights.

    { // Setup scene.
        if (model.scenes.size() > 1)
            printf("GLTFLoader::load warning: Only one scene supported. The default scene will be imported\n");

        // No scene loaded.
        if (model.defaultScene < 0)
            SceneNodes::UID::invalid_UID();

        const tinygltf::Scene& scene = model.scenes[model.defaultScene];
        if (scene.nodes.size() == 1) {
            // Only one node. Import it and return.
            const auto& node = model.nodes[scene.nodes[0]];
            return import_node(model, node, Transform::identity(), loaded_meshes_start_index, loaded_meshes, loaded_material_IDs);
        } else {
            // Several root nodes in the scene. Attach them to a single common root node.
            SceneNode root_node = SceneNodes::create("Scene root");
            for (size_t i = 0; i < scene.nodes.size(); i++) {
                const auto& node = model.nodes[scene.nodes[i]];
                SceneNode child_node = import_node(model, node, root_node.get_global_transform(), loaded_meshes_start_index, loaded_meshes, loaded_material_IDs);
                child_node.set_parent(root_node);
            }
            return root_node.get_ID();
        }
    }
    return SceneNodes::UID::invalid_UID();
}

bool file_supported(const std::string& filename) {
    return string_ends_with(filename, ".glb") || string_ends_with(filename, ".gltf");
}

} // NS GLTFLoader
