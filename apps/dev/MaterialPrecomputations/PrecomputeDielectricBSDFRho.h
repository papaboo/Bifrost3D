// Precompute the directional-hemispherical reflectance function (rho).
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _PRECOMPUTE_DIELECTRIC_BSDF_RHO_H_
#define _PRECOMPUTE_DIELECTRIC_BSDF_RHO_H_

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Math/RNG.h>

#include <fstream>

namespace PrecomputeDielectricBSDFRho {

using namespace Bifrost;
using namespace Bifrost::Math;
using namespace optix;
using namespace OptiXRenderer;

// Using the same specularity (or F0) range of [0.0125, 0.25] as in
// Enforcing Energy Preservation in Microfacet Models, Sforza et al,
const float ior_i_offset = 0.01666667f; // Slight offset to give perfect results for ior_i 1.5, which is the most representative ior value.
const float min_dense_ior_i = 1.25f + ior_i_offset; // Relative IOR of medium with specularity 0.0125 and traversed through air.
const float max_dense_ior_i = 3.0f + ior_i_offset; // Relative IOR of medium with specularity 0.25 and traversed through air.
const float min_light_ior_i = 1.0f / max_dense_ior_i;
const float max_light_ior_i = 1.0f / min_dense_ior_i;

typedef BSDFSample(*SampleDieletricBSDF)(float roughness, float ior_i, float3 wo, float3 random_sample);

float2 sample_rho(float3 wo, float roughness, float ior_i, unsigned int sample_count, const Math::RNG::PmjbRNG& rng, SampleDieletricBSDF sample_rough_BSDF) {

    // Each hemisphere should receive at least half the number of samples expected by the input sample count.
    unsigned int sample_count_per_hemisphere = sample_count / 2;

    // Predraw 256 samples to estimate the distribution of samples between each hemisphere.
    int total_sample_count;
    {
        const int rng_pre_sample_count = 256;
        int reflected_sample_count = 0, transmitted_sample_count = 0;
        for (int s = 0; s < rng_pre_sample_count; ++s) {
            Vector3f uv = rng.sample3f(s, rng_pre_sample_count);
            BSDFSample sample = sample_rough_BSDF(roughness, ior_i, wo, { uv.x, uv.y, uv.z });
            if (sample.PDF.is_valid()) {
                if (sample.direction.z < 0.0f)
                    ++transmitted_sample_count;
                else
                    ++reflected_sample_count;
            }
        }

        float valid_sample_count = float(transmitted_sample_count + reflected_sample_count);
        float reflected_ratio = reflected_sample_count / valid_sample_count;
        float transmitted_ratio = transmitted_sample_count / valid_sample_count;

        float smallest_ratio = fminf(reflected_ratio, transmitted_ratio);
        if (smallest_ratio == 0.0f)
            smallest_ratio = 0.5f;
        total_sample_count = int(sample_count_per_hemisphere / smallest_ratio);

        // Since we use a QMC RNG the sample count should be a power of two.
        total_sample_count = next_power_of_two(total_sample_count);
        total_sample_count = min(total_sample_count, (int)rng.m_max_sample_capacity);
    }

    double reflected_throughput = 0.0;
    double transmitted_throughput = 0.0;
    for (int s = 0; s < total_sample_count; ++s) {
        Vector3f uv = rng.sample3f(s, total_sample_count);
        BSDFSample sample = sample_rough_BSDF(roughness, ior_i, wo, { uv.x, uv.y, uv.z });
        if (sample.PDF.is_valid()) {
            double& throughput = sample.direction.z < 0.0f ? transmitted_throughput : reflected_throughput;
            throughput += sample.reflectance.x * abs(sample.direction.z) / sample.PDF.value();
        }
    }

    double reflected_rho = reflected_throughput / total_sample_count;
    double transmitted_rho = transmitted_throughput / total_sample_count;
    return { float(reflected_rho), float(transmitted_rho) };
}

struct TabulatedRho {
    float min_light_ior_i, max_light_ior_i;
    Assets::Image into_light_medium;
    float min_dense_ior_i, max_dense_ior_i;
    Assets::Image into_dense_medium;
};

TabulatedRho tabulate_rho(int width, int height, int depth, unsigned int sample_count, const Math::RNG::PmjbRNG& rng, SampleDieletricBSDF sample_rough_BSDF) {
    Assets::Image rho_into_light_medium_image = Assets::Image::create3D("rho into light medium", Assets::PixelFormat::RGB_Float, false, Vector3ui(width, height, depth));
    RGB* rho_into_light_medium_pixels = rho_into_light_medium_image.get_pixels<RGB>();
    Assets::Image rho_into_dense_medium_image = Assets::Image::create3D("rho into dense medium", Assets::PixelFormat::RGB_Float, false, Vector3ui(width, height, depth));
    RGB* rho_into_dense_medium_pixels = rho_into_dense_medium_image.get_pixels<RGB>();

    #pragma omp parallel for
    for (int z = 0; z < depth; ++z) {
        float ior_t = z / float(depth - 1);
        float light_ior_i = lerp(min_light_ior_i, max_light_ior_i, ior_t);
        float dense_ior_i = lerp(min_dense_ior_i, max_dense_ior_i, ior_t);
        for (int y = 0; y < height; ++y) {
            float roughness = y / float(height - 1);
            for (int x = 0; x < width; ++x) {
                float cos_theta = fmaxf(0.000001f, x / float(width - 1));
                float3 wo = make_float3(sqrt(1.0f - cos_theta * cos_theta), 0.0f, cos_theta);

                float2 light_rho = sample_rho(wo, roughness, light_ior_i, sample_count, rng, sample_rough_BSDF);
                rho_into_light_medium_pixels[x + width * (y + z * height)] = RGB(light_rho.x + light_rho.y, light_rho.x, light_rho.y);

                float2 dense_rho = sample_rho(wo, roughness, dense_ior_i, sample_count, rng, sample_rough_BSDF);
                rho_into_dense_medium_pixels[x + width * (y + z * height)] = RGB(dense_rho.x + dense_rho.y, dense_rho.x, dense_rho.y);
            }
        }
    }

    TabulatedRho result;
    result.min_light_ior_i = min_light_ior_i;
    result.max_light_ior_i = max_light_ior_i;
    result.into_light_medium = rho_into_light_medium_image;
    result.min_dense_ior_i = min_dense_ior_i;
    result.max_dense_ior_i = max_dense_ior_i;
    result.into_dense_medium = rho_into_dense_medium_image;
    return result;
}

std::string format_float(float v) {
    std::ostringstream out;
    out << v;
    if (out.str().length() == 1)
        out << ".0f";
    else
        out << "f";
    return out.str();
}

void output_brdf(TabulatedRho rho, const std::string& filename, const std::string& data_name, const std::string& description) {

    unsigned int cos_angle_sample_count = rho.into_light_medium.get_width();
    unsigned int roughness_sample_count = rho.into_light_medium.get_height();
    unsigned int ior_i_sample_count = rho.into_light_medium.get_depth();
    RGB* into_light_medium_pixels = rho.into_light_medium.get_pixels<RGB>();
    RGB* into_dense_medium_pixels = rho.into_dense_medium.get_pixels<RGB>();

    float min_IOR_into_light_medium = rho.min_light_ior_i;
    float max_IOR_into_light_medium = rho.max_light_ior_i;
    float IOR_into_light_medium_range = max_IOR_into_light_medium - min_IOR_into_light_medium;
    float min_IOR_into_dense_medium = rho.min_dense_ior_i;
    float max_IOR_into_dense_medium = rho.max_dense_ior_i;
    float IOR_into_dense_medium_range = max_IOR_into_dense_medium - min_IOR_into_dense_medium;

    std::ofstream out_header(filename);
    out_header <<
        "// " << description << "\n"
        "// ------------------------------------------------------------------------------------------------\n"
        "// Copyright (C) Bifrost. See AUTHORS.txt for authors\n"
        "//\n"
        "// This program is open source and distributed under the New BSD License.\n"
        "// See LICENSE.txt for more detail.\n"
        "// ------------------------------------------------------------------------------------------------\n"
        "// Generated by MaterialPrecomputations application.\n"
        "// ------------------------------------------------------------------------------------------------\n"
        "\n"
        "#include <Bifrost/Assets/Shading/Fittings.h>\n"
        "#include <Bifrost/Math/ImageSampling.h>\n"
        "#include <Bifrost/Math/Vector.h>\n"
        "\n"
        "namespace Bifrost::Assets::Shading::Rho {\n"
        "\n"
        "const int " << data_name << "_angle_sample_count = " << cos_angle_sample_count << "u;\n"
        "const int " << data_name << "_roughness_sample_count = " << roughness_sample_count << "u;\n"
        "const int " << data_name << "_ior_i_over_o_sample_count = " << ior_i_sample_count << "u;\n"
        "const float " << data_name << "_minimum_IOR_into_light_medium = " << format_float(min_IOR_into_light_medium) << ";\n"
        "const float " << data_name << "_maximum_IOR_into_light_medium = " << format_float(max_IOR_into_light_medium) << ";\n"
        "const float " << data_name << "_minimum_IOR_into_dense_medium = " << format_float(min_IOR_into_dense_medium) << ";\n"
        "const float " << data_name << "_maximum_IOR_into_dense_medium = " << format_float(max_IOR_into_dense_medium) << ";\n"
        "\n";

    // Output rho values for light medium
    out_header << "const Math::Vector2f " << data_name << "_into_light_medium[] = {\n";
    for (int z = 0; z < int(ior_i_sample_count); ++z) {
        float specularity_t = z / float(ior_i_sample_count - 1);
        float ior_i = lerp(min_IOR_into_light_medium, max_IOR_into_light_medium, specularity_t);
        out_header << "    // IOR_i " << ior_i << "\n";
        for (int y = 0; y < int(roughness_sample_count); ++y) {
            float roughness = y / float(roughness_sample_count - 1);
            out_header << "    // Roughness " << roughness << "\n";
            out_header << "    ";
            for (int x = 0; x < int(cos_angle_sample_count); ++x) {
                RGB rho = into_light_medium_pixels[x + y * cos_angle_sample_count + z * cos_angle_sample_count * roughness_sample_count];
                float total_reflectance = rho.r;
                float reflected_reflectance = rho.g;
                out_header << "{" << format_float(total_reflectance) << ", " << format_float(reflected_reflectance) << "}, ";
            }
            out_header << "\n";
        }
    }
    out_header << "};\n\n";

    // Output rho values for dense medium
    out_header << "const Math::Vector2f " << data_name << "_into_dense_medium[] = {\n";
    for (int z = 0; z < int(ior_i_sample_count); ++z) {
        float specularity_t = z / float(ior_i_sample_count - 1);
        float ior_i = lerp(min_IOR_into_dense_medium, max_IOR_into_dense_medium, specularity_t);
        out_header << "    // IOR_i " << ior_i << "\n";
        for (int y = 0; y < int(roughness_sample_count); ++y) {
            float roughness = y / float(roughness_sample_count - 1);
            out_header << "    // Roughness " << roughness << "\n";
            out_header << "    ";
            for (int x = 0; x < int(cos_angle_sample_count); ++x) {
                RGB rho = into_dense_medium_pixels[x + y * cos_angle_sample_count + z * cos_angle_sample_count * roughness_sample_count];
                float total_reflectance = rho.r;
                float reflected_reflectance = rho.g;
                out_header << "{" << format_float(total_reflectance) << ", " << format_float(reflected_reflectance) << "}, ";
            }
            out_header << "\n";
        }
    }
    out_header << "};\n\n";

    out_header <<
        "DielectricRho sample_light_medium_" << data_name << "(float wo_dot_normal, float roughness, float ior_i_over_o) {\n"
        "    // ior_i_over_o is below 1 when transitioning from a higher density medium to a lighter density one.\n"
        "    float ior_i_over_o_t = (ior_i_over_o - " << min_IOR_into_light_medium << "f) / " << IOR_into_light_medium_range << "f; // Remap valid ior_i_over_o values from [" << min_IOR_into_light_medium << ", " << max_IOR_into_light_medium << "] to [0, 1]\n"
        "    Math::Vector2f rho = Math::ImageSampling::trilinear(" << data_name << "_into_light_medium,\n"
        "        " << data_name << "_angle_sample_count, " << data_name << "_roughness_sample_count, " << data_name << "_ior_i_over_o_sample_count,\n"
        "        wo_dot_normal, roughness, ior_i_over_o_t);\n"
        "    return { rho.x, rho.y };\n"
        "}\n"
        "\n"
        "DielectricRho sample_dense_medium_" << data_name << "(float wo_dot_normal, float roughness, float ior_i_over_o) {\n"
        "    // ior_i_over_o is above 1 when transitioning from a lower density medium to a higher density one.\n"
        "    float ior_i_over_o_t = (ior_i_over_o - " << min_IOR_into_dense_medium << "f) / " << IOR_into_dense_medium_range << "f; // Remap valid ior_i_over_o values from [" << min_IOR_into_dense_medium << ", " << max_IOR_into_dense_medium << "] to [0, 1]\n"
        "    Math::Vector2f rho = Math::ImageSampling::trilinear(" << data_name << "_into_dense_medium,\n"
        "        " << data_name << "_angle_sample_count, " << data_name << "_roughness_sample_count, " << data_name << "_ior_i_over_o_sample_count,\n"
        "        wo_dot_normal, roughness, ior_i_over_o_t);\n"
        "    return { rho.x, rho.y };\n"
        "}\n"
        "\n"
        "DielectricRho sample_" << data_name << "(float wo_dot_normal, float roughness, float ior_i_over_o) {\n"
        "   if (ior_i_over_o < 1.0f)\n"
        "       return sample_light_medium_" << data_name << "(wo_dot_normal, roughness, ior_i_over_o);\n"
        "   else\n"
        "       return sample_dense_medium_" << data_name << "(wo_dot_normal, roughness, ior_i_over_o);\n"
        "}\n"
        "} // NS Bifrost::Assets::Shading::Rho\n";

    out_header.close();
}

} // NS PrecomputeRoughBRDFRho

#endif // _PRECOMPUTE_DIELECTRIC_BSDF_RHO_H_