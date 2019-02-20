// Bifrost glTF model loader.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#pragma warning(disable : 4267) // Implicit conversion from size_t to int in tiny_glTF.h

#include <glTFLoader/glTFLoader.h>

#include <Bifrost/Assets/Material.h>
#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Math/Conversions.h>

#include <StbImageLoader/StbImageLoader.h>

#define TINYGLTF_IMPLEMENTATION 1
#define TINYGLTF_NO_STB_IMAGE 1
#define TINYGLTF_NO_STB_IMAGE_WRITE 1
#include <glTFLoader/tiny_gltf.h>

#include <unordered_map>

using namespace Bifrost::Assets;
using namespace Bifrost::Core;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;

using Matrix3x3d = Matrix3x3<double>;

namespace glTFLoader {

// ------------------------------------------------------------------------------------------------
// Utils.
// ------------------------------------------------------------------------------------------------

inline bool string_ends_with(const std::string& s, const std::string& end) {
    if (s.length() < end.length())
        return false;

    return s.compare(s.length() - end.length(), end.length(), end) == 0;
}

struct RGBA32 { unsigned char r, g, b, a; };

// ------------------------------------------------------------------------------------------------
// Texture sampler conversion utils.
// ------------------------------------------------------------------------------------------------

inline MagnificationFilter convert_magnification_filter(int glTF_magnification_filter) {
    bool is_nearest = glTF_magnification_filter == TINYGLTF_TEXTURE_FILTER_NEAREST;
    return is_nearest ? MagnificationFilter::None : MagnificationFilter::Linear;
}

inline MinificationFilter convert_minification_filter(int glTF_minification_filter) {
    switch (glTF_minification_filter) {
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
        printf("glTFLoader::load warning: Unknown minification filter mode %u.\n", glTF_minification_filter);
        return MinificationFilter::Trilinear;
    }
}

inline WrapMode convert_wrap_mode(int glTF_wrap_mode) {
    if (glTF_wrap_mode == TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE)
        return WrapMode::Clamp;
    else if (glTF_wrap_mode == TINYGLTF_TEXTURE_WRAP_REPEAT)
        return WrapMode::Repeat;
    else if (glTF_wrap_mode == TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT)
        printf("glTFLoader::load error: Mirrored repeat wrap mode not supported.\n");
    return WrapMode::Repeat;
}

// ------------------------------------------------------------------------------------------------
// Image conversion utils.
// Needed as glTF stores (tint, coverage) in one image and (metallic, roughness) in another.
// Bifrost recognises that the most used maps are tint and roughness, so those are stored in a 
// single (tint, roughness) image and metallic and coverage are stored in two seperate images.
// ------------------------------------------------------------------------------------------------

enum class ImageUsage { Tint = 1, Coverage = 2, Metallic = 4, Roughness = 8, Tint_roughness = 9 };

inline std::string to_string(ImageUsage usage) {
    switch (usage) {
    case ImageUsage::Tint: return "tint";
    case ImageUsage::Coverage: return "coverage";
    case ImageUsage::Metallic: return "metallic";
    case ImageUsage::Roughness: return "roughness";
    case ImageUsage::Tint_roughness: return "tint_roughness";
    default: return "unknown";
    };
}

typedef size_t ConvertedImageKey;
inline size_t hash_converted_image(Images::UID image_ID, Images::UID image2_ID, ImageUsage usage) {
    return (size_t)image2_ID.get_index() << 32 | image_ID.get_index() << 8 | (size_t)usage;
}
inline size_t hash_converted_image(Images::UID image_ID, ImageUsage usage) {
    return hash_converted_image(image_ID, Images::UID::invalid_UID(), usage);
}

using ImageCache = std::unordered_map<ConvertedImageKey, Images::UID>;

struct SamplerParams {
    MagnificationFilter Magnification_filter = MagnificationFilter::Linear;
    MinificationFilter Minification_filter = MinificationFilter::Trilinear;
    WrapMode WrapU = WrapMode::Repeat;
    WrapMode WrapV = WrapMode::Repeat;

    void parse_glTF_sampler(tinygltf::Sampler glTF_sampler) {
        Magnification_filter = convert_magnification_filter(glTF_sampler.magFilter);
        Minification_filter = convert_minification_filter(glTF_sampler.minFilter);
        WrapU = convert_wrap_mode(glTF_sampler.wrapR);
        WrapV = convert_wrap_mode(glTF_sampler.wrapS);
    }

    inline Textures::UID create_texture_2D(Images::UID image_ID) {
        if (Images::has(image_ID))
            return Textures::create2D(image_ID, Magnification_filter, Minification_filter, WrapU, WrapV);
        return Textures::UID::invalid_UID();
    }
};

struct TextureState {
    int glTF_image_index = -1;
    Images::UID Image_ID = Images::UID::invalid_UID();
    SamplerParams Sampler;

    void parse_glTF_texture(tinygltf::Model& model, int texture_index) {
        const auto& glTF_texture = model.textures[texture_index];
        glTF_image_index = glTF_texture.source;
        const auto& glTF_image = model.images[glTF_image_index];
        memcpy(&Image_ID, glTF_image.image.data(), sizeof(Image_ID));
        assert(Images::has(Image_ID));

        // Extract sampler params.
        if (glTF_texture.sampler >= 0)
            Sampler.parse_glTF_sampler(model.samplers[glTF_texture.sampler]);
    }
};

// Extracts a single channel from an image or retrieves it from the cache. The resulting image is cached.
Image extract_channel(Image image, int channel, ImageUsage usage, const std::string& name, ImageCache& converted_images) {
    if (channel >= channel_count(image.get_pixel_format()))
        return Images::UID::invalid_UID();

    if (image.get_pixel_format() == PixelFormat::A8 && channel == 0)
        return image;

    auto converted_image_hash = hash_converted_image(image.get_ID(), usage);
    auto& converted_image_itr = converted_images.find(converted_image_hash);
    if (converted_image_itr != converted_images.end())
        return converted_image_itr->second;

    // Extract the channel from the image and store the resulting image in the cache.
    unsigned int mipmap_count = image.get_mipmap_count();
    Vector2ui size = Vector2ui(image.get_width(), image.get_height());
    Images::UID single_channel_image_ID = Images::create2D(name + "_" + to_string(usage), PixelFormat::A8, 1.0, size, mipmap_count);

    unsigned char min_value = 255;
    for (unsigned int m = 0; m < mipmap_count; ++m) {
        int pixel_count = image.get_pixel_count(m);
        unsigned char* dst_pixels = Images::get_pixels<unsigned char>(single_channel_image_ID, m);

        auto src_format = image.get_pixel_format();
        if (src_format == PixelFormat::RGB24 || src_format == PixelFormat::RGBA32) {
            // Memcpy unsigned char channels.
            unsigned char* src_pixels = ((unsigned char*)image.get_pixels(m)) + channel;
            int src_pixel_stride = size_of(image.get_pixel_format());

            for (int p = 0; p < pixel_count; ++p) {
                dst_pixels[p] = *src_pixels;
                min_value = min(min_value, dst_pixels[p]);
                src_pixels += src_pixel_stride;
            }
        } else {
            // Copy using get pixel and data conversion.
            for (int p = 0; p < pixel_count; ++p) {
                RGBA pixel = image.get_pixel(p, m);
                dst_pixels[p] = unsigned char(pixel[channel] * 255 + 0.5f);
                min_value = min(min_value, dst_pixels[p]);
            }
        }
    }

    // Only create single channel image if it has values other than 1.0.
    if (min_value < 255) {
        converted_images.insert({ converted_image_hash, single_channel_image_ID });
        return single_channel_image_ID;
    } else {
        Images::destroy(single_channel_image_ID);
        converted_images.insert({ converted_image_hash, Images::UID::invalid_UID() });
        return Images::UID::invalid_UID();
    }
}

Image extract_tint_roughness(Image tint_image, Image roughness_image, const std::string& name, ImageCache& converted_images) {
    // Handle case where tint doesn't exist.
    if (!tint_image.exists())
        return extract_channel(roughness_image, 1, ImageUsage::Roughness, name, converted_images);

    // Return tint_image if it has three channels and roughness image doesn't exist.
    if (!roughness_image.exists() && channel_count(tint_image.get_pixel_format()) == 3)
        return tint_image;

    auto tint_roughness_image_hash = hash_converted_image(tint_image.get_ID(), roughness_image.get_ID(), ImageUsage::Tint_roughness);
    auto& converted_image_itr = converted_images.find(tint_roughness_image_hash);
    if (converted_image_itr != converted_images.end())
        return converted_image_itr->second;

    unsigned int mipmap_count = tint_image.get_mipmap_count();
    Vector2ui size = Vector2ui(tint_image.get_width(), tint_image.get_height());
    Image dst_image = Images::create2D(name + "_tint_roughness", PixelFormat::RGBA32, tint_image.get_gamma(), size, mipmap_count);

    // Fill tint channels
    assert(tint_image.get_pixel_format() == PixelFormat::RGB24 || tint_image.get_pixel_format() == PixelFormat::RGBA32);
    for (unsigned int m = 0; m < mipmap_count; ++m) {
        unsigned char* src_pixels = (unsigned char*)tint_image.get_pixels(m);
        int src_pixel_size = size_of(tint_image.get_pixel_format());
        RGBA32* dst_pixels = dst_image.get_pixels<RGBA32>(m);

        int pixel_count = tint_image.get_pixel_count(m);
        for (int p = 0; p < pixel_count; ++p) {
            dst_pixels[p].r = src_pixels[0];
            dst_pixels[p].g = src_pixels[1];
            dst_pixels[p].b = src_pixels[2];
            dst_pixels[p].a = 255;
            src_pixels += src_pixel_size;
        }
    }

    // Fill roughness channels. NOTE Gamma correct?
    if (roughness_image.exists() && channel_count(roughness_image.get_pixel_format()) >= 2) {
        auto roughness_pixel_format = roughness_image.get_pixel_format();
        // Assert that image dimensions of the two images are equal and that the roughness image has 8bit channels.
        assert(tint_image.get_width() == roughness_image.get_width());
        assert(tint_image.get_height() == roughness_image.get_height());
        assert(tint_image.get_mipmap_count() == roughness_image.get_mipmap_count());
        assert(roughness_pixel_format == PixelFormat::RGB24 || roughness_pixel_format == PixelFormat::RGBA32);
        for (unsigned int m = 0; m < mipmap_count; ++m) {
            unsigned char* src_pixels = (unsigned char*)roughness_image.get_pixels(m);
            int src_pixel_size = size_of(roughness_image.get_pixel_format());
            RGBA32* dst_pixels = dst_image.get_pixels<RGBA32>(m);

            int pixel_count = tint_image.get_pixel_count(m);
            for (int p = 0; p < pixel_count; ++p) {
                dst_pixels[p].a = src_pixels[1];
                src_pixels += src_pixel_size;
            }
        }
    }

    converted_images.insert({ tint_roughness_image_hash, dst_image.get_ID() });
    return dst_image;
}

// ------------------------------------------------------------------------------------------------
// Importer
// ------------------------------------------------------------------------------------------------

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

    // X-coord is negated as glTF uses a right-handed coordinate system and Bifrost a right-handed.
    // This also means that we have to invert some of the entries in the quaternion, specifically .x and the rotation direction in .w.
    local_transform.rotation.x = -local_transform.rotation.x;
    local_transform.rotation.w = -local_transform.rotation.w;
    local_transform.translation.x = -local_transform.translation.x;

    Transform global_transform = parent_transform * local_transform;
    SceneNodes::UID scene_node_ID = SceneNodes::create(node.name, global_transform);

    if (node.mesh >= 0) {
        // Get loaded mesh indices.
        int glTF_mesh_index = node.mesh;
        int mesh_index = meshes_start_index[glTF_mesh_index];
        int mesh_end_index = meshes_start_index[glTF_mesh_index + 1];

        for (const auto& primitive : model.meshes[glTF_mesh_index].primitives) {
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
                printf("GLTFLoader::load warning: %s primitive %u not supported.\n", model.meshes[glTF_mesh_index].name.c_str(), primitive.mode);
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
void copy_indices(unsigned int* bifrost_indices, const T* glTF_indices, size_t count, int index_byte_stride) {
    int glTF_index_T_stride = index_byte_stride / sizeof(T);
    for (size_t i = 0; i < count; ++i) {
        *bifrost_indices = *glTF_indices;
        ++bifrost_indices;
        glTF_indices += glTF_index_T_stride;
    }
}

// ------------------------------------------------------------------------------------------------
// Loads a glTF file.
// ------------------------------------------------------------------------------------------------
SceneNodes::UID load(const std::string& filename) {

    // See https://github.com/syoyo/tinygltf/blob/master/loader_example.cc

    tinygltf::Model model;
    tinygltf::TinyGLTF glTF_ctx;
    std::string errors, warnings;

    static auto image_loader = [](tinygltf::Image* glTF_image, std::string* error, std::string* warning, int req_width, int req_height,
                                  const unsigned char *bytes, int size, void* user_data) -> bool {

        Image image = StbImageLoader::load_from_memory("Unnamed", bytes, size);

        if (!image.exists()) {
            if (error)
                *error += "glTFLoader::load error: Failed to load image from memory.\n";
            return false;
        }

        if (image.get_width() < 1 || image.get_height() < 1) {
            Images::destroy(image.get_ID());
            if (error)
                *error += "glTFLoader::load error: Failed to load image from memory.\n";
            return false;
        }

        if (req_width > 0 && image.get_width() != req_width) {
            Images::destroy(image.get_ID());
            if (error)
                *error += "glTFLoader::load error: Image width mismatch.\n";
            return false;
        }

        if (req_height > 0 && image.get_height() != req_height) {
            Images::destroy(image.get_ID());
            if (error)
                *error += "glTFLoader::load error: Image height mismatch.\n";
            return false;
        }

        glTF_image->width = image.get_width();
        glTF_image->height = image.get_height();
        glTF_image->component = channel_count(image.get_pixel_format());
        // HACK Store image ID in pixels instead of pixel data.
        glTF_image->image.resize(4);
        memcpy(glTF_image->image.data(), &image.get_ID(), sizeof(Images::UID));

        return true;
    };

    glTF_ctx.SetImageLoader(image_loader, nullptr);

    bool ret = false;
    if (string_ends_with(filename, "glb"))
        ret = glTF_ctx.LoadBinaryFromFile(&model, &errors, &warnings, filename.c_str());
    else if (string_ends_with(filename, "gltf"))
        ret = glTF_ctx.LoadASCIIFromFile(&model, &errors, &warnings, filename.c_str());
    else {
        printf("glTFLoader::load error: '%s' not a glTF file\n", filename.c_str());
        return SceneNodes::UID::invalid_UID();
    }

    if (!warnings.empty())
        printf("glTFLoader::load warning: %s\n", warnings.c_str());


    if (!errors.empty())
        printf("glTFLoader::load error: %s\n", errors.c_str());

    if (!ret) {
        printf("glTFLoader::load error: Failed to parse '%s'\n", filename.c_str());
        return SceneNodes::UID::invalid_UID();
    }

    // Import materials.
    ImageCache converted_images;
    auto image_is_used = std::vector<bool>(model.images.size());
    std::fill(image_is_used.begin(), image_is_used.end(), false);
    auto loaded_material_IDs = std::vector<Materials::UID>(model.materials.size());

    auto flag_image_as_used = [&](int glTF_image_index, Images::UID glTF_image, Images::UID converted_image) {
        if (glTF_image_index >= 0 && glTF_image == converted_image)
            image_is_used[glTF_image_index] = true;
    };

    for (int i = 0; i < model.materials.size(); ++i) {
        Materials::Data mat_data = {};
        mat_data.tint = RGB::white();
        mat_data.roughness = 1.0f;
        mat_data.metallic = 0.0f;
        mat_data.coverage = 1.0f;
        bool is_opaque = true;

        const auto& glTF_mat = model.materials[i];
        // Process additional values first, as we need to know the alphaMode before converting images.
        for (const auto& val : glTF_mat.additionalValues) {
            if (val.first.compare("doubleSided") == 0)
                // NOTE to self: Double sided should set a 'thin/doubleSided' property on the meshes instead of on the materials.
                // In case the mesh is used by one sided and two sided meshes, we need to duplicate the mesh.
                printf("GLTFLoader::load warning: doubleSided property not supported.\n");
            else if (val.first.compare("alphaMode") == 0) {
                is_opaque = false;
                bool is_cutout = val.second.string_value.compare("MASK") == 0;
                mat_data.flags |= is_cutout ? MaterialFlag::Cutout : MaterialFlag::None;
            }
        }

        TextureState tint_coverage_tex;
        TextureState metallic_roughness_tex;

        for (const auto& val : glTF_mat.values) {
            if (val.first.compare("baseColorFactor") == 0) {
                auto tint = val.second.number_array;
                mat_data.tint = { float(tint[0]), float(tint[1]), float(tint[2]) };
            } else if (val.first.compare("baseColorTexture") == 0)
                tint_coverage_tex.parse_glTF_texture(model, val.second.TextureIndex());
            else if (val.first.compare("metallicRoughnessTexture") == 0)
                metallic_roughness_tex.parse_glTF_texture(model, val.second.TextureIndex());
            else if (val.first.compare("roughnessFactor") == 0)
                mat_data.roughness = float(val.second.Factor());
            else if (val.first.compare("metallicFactor") == 0)
                mat_data.metallic = float(val.second.Factor());
        }

        // Convert images from glTF channel layout to Bifrost channel layout.
        auto metallic_image = extract_channel(metallic_roughness_tex.Image_ID, 2, ImageUsage::Metallic, glTF_mat.name, converted_images);
        flag_image_as_used(metallic_roughness_tex.glTF_image_index, metallic_roughness_tex.Image_ID, metallic_image.get_ID());
        mat_data.metallic_texture_ID = metallic_roughness_tex.Sampler.create_texture_2D(metallic_image.get_ID());

        auto coverage_image = extract_channel(tint_coverage_tex.Image_ID, 3, ImageUsage::Coverage, glTF_mat.name, converted_images);
        flag_image_as_used(tint_coverage_tex.glTF_image_index, tint_coverage_tex.Image_ID, coverage_image.get_ID());
        mat_data.coverage_texture_ID = tint_coverage_tex.Sampler.create_texture_2D(coverage_image.get_ID());

        auto tint_roughness_image = extract_tint_roughness(tint_coverage_tex.Image_ID, metallic_roughness_tex.Image_ID, glTF_mat.name, converted_images);
        flag_image_as_used(tint_coverage_tex.glTF_image_index, tint_coverage_tex.Image_ID, tint_roughness_image.get_ID());
        flag_image_as_used(tint_coverage_tex.glTF_image_index, metallic_roughness_tex.Image_ID, tint_roughness_image.get_ID());
        mat_data.tint_roughness_texture_ID = tint_coverage_tex.Sampler.create_texture_2D(tint_roughness_image.get_ID());

        loaded_material_IDs[i] = Materials::create(glTF_mat.name, mat_data);
    }

    // Delete images not used by the datamodel.
    for (int i = 0; i < image_is_used.size(); ++i) {
        if (!image_is_used[i]) {
            const auto& glTF_image = model.images[i];
            Images::UID image_ID;
            memcpy(&image_ID, glTF_image.image.data(), sizeof(image_ID));
            Images::destroy(image_ID);
        }
    }

    // Import models.
    auto loaded_meshes_start_index = std::vector<int>();
    loaded_meshes_start_index.reserve(model.meshes.size());
    auto loaded_meshes = std::vector<Mesh>();
    loaded_meshes.reserve(model.meshes.size());
    for (const auto& glTF_mesh : model.meshes) {
        // Mark at what index in loaded_meshes that the models corresponding to the current glTF mesh begins.
        loaded_meshes_start_index.push_back(loaded_meshes.size());
        for (size_t p = 0; p < glTF_mesh.primitives.size(); ++p) {
            const auto& primitive = glTF_mesh.primitives[p];
            
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
                printf("GLTFLoader::load warning: %s[%zu] primitive %u not supported.\n", glTF_mesh.name.c_str(), p, primitive.mode);
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

            Mesh mesh = Meshes::create(glTF_mesh.name, primitive_count, vertex_count, mesh_flags);
            
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

    // KHR_lights_cmn not supported.
    if (model.lights.size() > 0)
        printf("GLTFLoader::load warning: KHR_lights_cmn not supported. Light sources will be ignored.\n");

    { // Setup scene.
        if (model.scenes.size() > 1)
            printf("GLTFLoader::load warning: Only one scene supported. The default scene will be imported.\n");

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

} // NS glTFLoader
