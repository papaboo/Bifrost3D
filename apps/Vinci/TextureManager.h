// Vinci texture manager
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _VINCI_TEXTURE_MANAGER_H_
#define _VINCI_TEXTURE_MANAGER_H_

#include <Bifrost/Assets/Material.h>
#include <Bifrost/Assets/Texture.h>
#include <Bifrost/Math/RNG.h>

#include <StbImageLoader/StbImageLoader.h>

#include <filesystem>

class TextureManager {
public:
    explicit TextureManager(const std::filesystem::path& texture_directory) {
        using namespace Bifrost::Assets;
        using namespace Bifrost::Math;
        using namespace std;
        namespace fs = std::filesystem;

        // Early out if the directory specified is invalid.
        std::error_code error_code;
        if (!fs::exists(texture_directory, error_code) || !fs::is_directory(texture_directory, error_code)) {
            printf("Vinci warning: Failed to parse texture directory. '%ws'\n", texture_directory.c_str());
            if (error_code.value())
                printf("               %s\n", error_code.message().c_str());
            return;
        }

        auto load_texture = [](fs::path path) -> Textures::UID {
            Images::UID image_ID = StbImageLoader::load(path.string());
            return Textures::create2D(image_ID, MagnificationFilter::Linear, MinificationFilter::Trilinear);
        };

        auto ends_with = [](const std::string& str, const std::string& suffix) -> bool {
            return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
        };

        // Search for files whose name ends with _col
        for (const auto& entry : fs::recursive_directory_iterator(texture_directory)) {
            if (entry.is_regular_file()) {
                fs::path path = entry.path();

                fs::path directory = path.parent_path();
                string tint_stem = path.stem().string();
                string extension = path.extension().string();
                
                if (!ends_with(tint_stem, "_col"))
                    continue;

                // Load color texture.
                m_tints.push_back(load_texture(path));

                // Remove the _col suffix.
                tint_stem.resize(tint_stem.size() - 4);
                
                { // Load roughness if it exists
                    fs::path roughness_map_path = directory / (tint_stem + "_rgh" + path.extension().string());
                    if (fs::exists(roughness_map_path)) {
                        Image roughness = StbImageLoader::load(roughness_map_path.string());
                        if (roughness.get_pixel_format() != PixelFormat::Roughness8) {
                            Image prev_roughness = roughness;
                            roughness = ImageUtils::change_format(roughness.get_ID(), PixelFormat::Roughness8, 1.0f, [](RGBA pixel) -> RGBA { return RGBA(pixel.r, pixel.r, pixel.r, pixel.r); });
                            Images::destroy(prev_roughness.get_ID());
                        }
                        m_roughness.push_back(Textures::create2D(roughness.get_ID(), MagnificationFilter::Linear, MinificationFilter::Trilinear));
                    } else
                        m_roughness.push_back(Textures::UID::invalid_UID());
                }

                { // Combine albedo and roughness
                    int texture_index = int(m_tints.size() - 1);
                    Texture tint_tex = m_tints[texture_index];
                    Texture roughness_tex = m_roughness[texture_index];
                    if (tint_tex.exists() && roughness_tex.exists()) {
                        Image tint_roughness_image = ImageUtils::combine_tint_roughness(tint_tex.get_image(), roughness_tex.get_image(), 3);
                        auto tint_roughness_tex_ID = Textures::create2D(tint_roughness_image.get_ID(), MagnificationFilter::Linear, MinificationFilter::Trilinear);
                        m_tint_roughness.push_back(tint_roughness_tex_ID);
                    } else
                        m_tint_roughness.push_back(Textures::UID::invalid_UID());
                }

                { // Load opacity
                    fs::path opacity_map_path = directory / (tint_stem + "_mask" + path.extension().string());
                    if (fs::exists(opacity_map_path)) {
                        Image opacity = StbImageLoader::load(opacity_map_path.string());
                        if (opacity.get_pixel_format() != PixelFormat::Alpha8) {
                            Image prev_opacity = opacity;
                            opacity = ImageUtils::change_format(opacity.get_ID(), PixelFormat::Alpha8, 1.0f, [](RGBA pixel) -> RGBA { return RGBA(pixel.r, pixel.r, pixel.r, pixel.r); });
                            Images::destroy(prev_opacity.get_ID());
                        }
                        m_opacity.push_back(Textures::create2D(opacity.get_ID(), MagnificationFilter::Linear, MinificationFilter::Trilinear));
                    }  else
                        m_opacity.push_back(Textures::UID::invalid_UID());
                }
            }
        }
    }
    ~TextureManager() {
        using namespace Bifrost::Assets;

        auto destroy_texture_assets = [](std::vector<Textures::UID> texture_IDs) {
            for (Texture texture : texture_IDs) {
                Images::destroy(texture.get_image().get_ID());
                Textures::destroy(texture.get_ID());
            }
        };

        destroy_texture_assets(m_tints);
        destroy_texture_assets(m_roughness);
        destroy_texture_assets(m_tint_roughness);
        destroy_texture_assets(m_opacity);
    }

    inline Bifrost::Assets::Materials::UID generate_random_material(Bifrost::Math::RNG::LinearCongruential& rng) const {
        auto tint = Bifrost::Math::RGB(rng.sample1f(), rng.sample1f(), rng.sample1f());
        float roughness = rng.sample1f();
        auto material_data = Bifrost::Assets::Materials::Data::create_dielectric(tint, roughness, 0.5f);

        float texture_probability = 0.9f;
        if (rng.sample1f() < texture_probability && m_tints.size() > 0) {
            int texture_sample = int(rng.sample1f() * m_tints.size() - 0.5f);

            if (Bifrost::Assets::Textures::has(m_tint_roughness[texture_sample]))
                material_data.tint_roughness_texture_ID = m_tint_roughness[texture_sample];
            else
                material_data.tint_roughness_texture_ID = m_tints[texture_sample];

            material_data.coverage_texture_ID = m_opacity[texture_sample];
        }
        return Bifrost::Assets::Materials::create("Mat", material_data);
    }

private:

    std::vector<Bifrost::Assets::Textures::UID> m_tints;
    std::vector<Bifrost::Assets::Textures::UID> m_roughness;
    std::vector<Bifrost::Assets::Textures::UID> m_tint_roughness;
    std::vector<Bifrost::Assets::Textures::UID> m_opacity;
};

#endif // _VINCI_TEXTURE_MANAGER_H_