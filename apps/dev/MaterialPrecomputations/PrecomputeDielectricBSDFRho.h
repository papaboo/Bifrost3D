// Precompute the directional-hemispherical reflectance function (rho).
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _PRECOMPUTE_DIELECTRIC_BSDF_RHO_H_
#define _PRECOMPUTE_DIELECTRIC_BSDF_RHO_H_

#include <PmjbRNG.h>

#include <Bifrost/Assets/Image.h>

#include <fstream>

namespace PrecomputeDielectricBSDFRho {

using namespace Bifrost;
using namespace optix;
using namespace OptiXRenderer;

// Using the same specularity (or F0) range of [0.0125, 0.25] as in
// Enforcing Energy Preservation in Microfacet Models, Sforza et al, 2022
const float min_specularity = 0.0125f;
const float max_specularity = 0.25f;
const float specularity_range = max_specularity - min_specularity;

typedef BSDFSample(*SampleDieletricBSDF)(float roughness, float specularity, float3 wo, float3 random_sample);

float2 sample_rho(float3 wo, float roughness, float specularity, unsigned int sample_count, const PmjbRNG& rng, SampleDieletricBSDF sample_rough_BSDF) {

    // Each hemisphere should receive at least half the number of samples expected by the input sample count.
    unsigned int sample_count_per_hemisphere = sample_count / 2;

    // Predraw 256 samples to estimate the distribution of samples between each hemisphere.
    int total_sample_count;
    {
        const int rng_pre_sample_count = 256;
        int reflected_sample_count = 0, transmitted_sample_count = 0;
        for (int s = 0; s < rng_pre_sample_count; ++s) {
            float3 rng_sample = rng.sample_3f(s, rng_pre_sample_count);
            BSDFSample sample = sample_rough_BSDF(roughness, specularity, wo, rng_sample);
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
        total_sample_count = min(total_sample_count, rng.m_max_sample_capacity);
    }

    double reflected_throughput = 0.0;
    double transmitted_throughput = 0.0;
    for (int s = 0; s < total_sample_count; ++s) {
        float3 rng_sample = rng.sample_3f(s, total_sample_count);
        BSDFSample sample = sample_rough_BSDF(roughness, specularity, wo, rng_sample);
        if (sample.PDF.is_valid()) {
            double& throughput = sample.direction.z < 0.0f ? transmitted_throughput : reflected_throughput;
            throughput += sample.reflectance.x * abs(sample.direction.z) / sample.PDF.value();
        }
    }

    double reflected_rho = reflected_throughput / total_sample_count;
    double transmitted_rho = transmitted_throughput / total_sample_count;
    return { float(reflected_rho), float(transmitted_rho) };
}

Assets::Image tabulate_rho(int width, int height, int depth, unsigned int sample_count, const PmjbRNG& rng, SampleDieletricBSDF sample_rough_BSDF) {
    Assets::Image rho_image = Assets::Image::create3D("rho", Assets::PixelFormat::RGB_Float, false, Math::Vector3ui(width, height, depth));
    Math::RGB* rho_image_pixels = rho_image.get_pixels<Math::RGB>();

    #pragma omp parallel for
    for (int z = 0; z < depth; ++z) {
        float specularity_t = z / float(depth - 1);
        float specularity = lerp(min_specularity, max_specularity, specularity_t);
        for (int y = 0; y < height; ++y) {
            float roughness = y / float(height - 1);
            for (int x = 0; x < width; ++x) {
                float cos_theta = fmaxf(0.000001f, x / float(width - 1));
                float3 wo = make_float3(sqrt(1.0f - cos_theta * cos_theta), 0.0f, cos_theta);
                float2 rho = sample_rho(wo, roughness, specularity, sample_count, rng, sample_rough_BSDF);
                rho_image_pixels[x + width * (y + z * height)] = Math::RGB(rho.x + rho.y, rho.x, rho.y);
            }
        }
    }

    return rho_image;
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

void output_brdf(Assets::Image image, int sample_count, const std::string& filename, const std::string& data_name, const std::string& description) {

    unsigned int cos_angle_sample_count = image.get_width();
    unsigned int roughness_sample_count = image.get_height();
    unsigned int specularity_sample_count = image.get_depth();
    Math::RGB* image_pixels = image.get_pixels<Math::RGB>();

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
        "\n"
        "using Bifrost::Math::Vector2f;\n\n"
        "namespace Bifrost::Assets::Shading::Rho {\n"
        "\n"
        "const int " << data_name << "_sample_count = " << sample_count << "u;\n"
        "const int " << data_name << "_angle_sample_count = " << cos_angle_sample_count << "u;\n"
        "const int " << data_name << "_roughness_sample_count = " << roughness_sample_count << "u;\n"
        "const int " << data_name << "_specularity_sample_count = " << specularity_sample_count << "u;\n"
        "const float " << data_name << "_minimum_specularity = " << format_float(min_specularity) << ";\n"
        "const float " << data_name << "_maximum_specularity = " << format_float(max_specularity) << ";\n"
        "\n"
        "const Vector2f " << data_name << "[] = {\n";

    for (int z = 0; z < int(specularity_sample_count); ++z) {
        float specularity_t = z / float(specularity_sample_count - 1);
        float specularity = lerp(min_specularity, max_specularity, specularity_t);
        out_header << "    // Specularity " << specularity << "\n";
        for (int y = 0; y < int(roughness_sample_count); ++y) {
            float roughness = y / float(roughness_sample_count - 1);
            out_header << "    // Roughness " << roughness << "\n";
            out_header << "    ";
            for (int x = 0; x < int(cos_angle_sample_count); ++x) {
                Math::RGB rho = image_pixels[x + y * cos_angle_sample_count + z * cos_angle_sample_count * roughness_sample_count];
                float total_reflectance = rho.r;
                float reflected_reflectance = rho.g;
                out_header << "Vector2f(" << format_float(total_reflectance) << ", " << format_float(reflected_reflectance / total_reflectance) << "), ";
            }
            out_header << "\n";
        }
    }

    out_header <<
        "};\n"
        "\n"
        "Vector2f sample_" << data_name << "(float wo_dot_normal, float roughness, float specularity) {\n"
        "    float specularity_t = (specularity - " << min_specularity << "f) / " << specularity_range << "f; // Remap valid specularity values to [0, 1]\n"
        "    return Math::ImageSampling::trilinear(" << data_name << ",\n"
        "        " << data_name << "_angle_sample_count, " << data_name << "_roughness_sample_count, " << data_name << "_specularity_sample_count,\n"
        "        wo_dot_normal, roughness, specularity_t);\n"
        "}\n"
        "\n"
        "} // NS Bifrost::Assets::Shading::Rho\n";

    out_header.close();
}

} // NS PrecomputeRoughBRDFRho

#endif // _PRECOMPUTE_DIELECTRIC_BSDF_RHO_H_