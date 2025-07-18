// Precompute the directional-hemispherical reflectance function (rho).
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _PRECOMPUTE_ROUGH_BRDF_RHO_H_
#define _PRECOMPUTE_ROUGH_BRDF_RHO_H_

#include <OptiXRenderer/Shading/BSDFs/GGX.h>
#include <OptiXRenderer/RNG.h>

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Core/Array.h>
#include <Bifrost/Math/Utils.h>

#include <fstream>

namespace PrecomputeRoughBRDFRho {

using namespace Bifrost;
using namespace optix;
using namespace OptiXRenderer;

typedef BSDFSample(*SampleRoughBRDF)(float roughness, float3 wo, float2 random_sample);

double sample_rho(float3 wo, float roughness, unsigned int sample_count, SampleRoughBRDF sample_rough_BSDF) {

    Core::Array<double> throughput = Core::Array<double>(sample_count);
    for (unsigned int s = 0; s < sample_count; ++s) {
        float2 rng_sample = RNG::sample02(s);
        BSDFSample sample = sample_rough_BSDF(roughness, wo, rng_sample);
        if (sample.PDF.is_valid())
            throughput[s] = sample.reflectance.x * sample.direction.z / sample.PDF.value();
        else
            throughput[s] = 0.0;
    }

    return Math::sort_and_pairwise_summation(throughput.begin(), throughput.end()) / sample_count;
}

Assets::Image tabulate_rho(unsigned int width, unsigned int height, unsigned int sample_count, SampleRoughBRDF sample_rough_BSDF) {
    Assets::Image rho_image = Assets::Image::create2D("rho", Assets::PixelFormat::RGB_Float, false, Math::Vector2ui(width, height));
    Math::RGB* rho_image_pixels = rho_image.get_pixels<Math::RGB>();

#pragma omp parallel for
    for (int y = 0; y < int(height); ++y) {
        float roughness = y / float(height - 1);
        for (int x = 0; x < int(width); ++x) {
            float cos_theta = fmaxf(0.000001f, x / float(width - 1));
            float3 wo = make_float3(sqrt(1.0f - cos_theta * cos_theta), 0.0f, cos_theta);
            double rho = sample_rho(wo, roughness, sample_count, sample_rough_BSDF);
            rho_image_pixels[x + y * width] = Math::RGB(float(rho));
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

template <int ElementDimensions>
void output_brdf(Assets::Image image, int sample_count, const std::string& filename, const std::string& data_name, const std::string& description) {

    unsigned int width = image.get_width();
    unsigned int height = image.get_height();
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
        "namespace Bifrost::Assets::Shading::Rho {\n"
        "\n"
        "const int " << data_name << "_sample_count = " << sample_count << "u;\n"
        "const int " << data_name << "_angle_sample_count = " << width << "u;\n"
        "const int " << data_name << "_roughness_sample_count = " << height << "u;\n"
        "\n"
        "const float " << data_name << "[] = {\n";

    for (int y = 0; y < int(height); ++y) {
        float roughness = y / float(height - 1);
        out_header << "    // Roughness " << roughness << "\n";
        out_header << "    ";
        for (int x = 0; x < int(width); ++x) {
            Math::RGB& rho = image_pixels[x + y * width];
            out_header << format_float(rho.r) << ", ";
        }
        out_header << "\n";
    }

    out_header <<
        "};\n"
        "\n"
        "float sample_" << data_name << "(float wo_dot_normal, float roughness) {\n"
        "    return Math::ImageSampling::bilinear(" << data_name << ", " << data_name << "_angle_sample_count, " << data_name << "_roughness_sample_count, wo_dot_normal, roughness);\n"
        "}\n"
        "\n"
        "} // NS Bifrost::Assets::Shading::Rho\n";

    out_header.close();
}

} // NS PrecomputeRoughBRDFRho

#endif // _PRECOMPUTE_ROUGH_BRDF_RHO_H_