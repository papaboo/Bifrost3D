// Precomputation of lookup tables for various BSDFs.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <FitRhoApproximation.h>

#include <OptiXRenderer/Shading/BSDFs/Burley.h>
#include <OptiXRenderer/Shading/BSDFs/GGX.h>
#include <OptiXRenderer/Shading/BSDFs/OrenNayar.h>
#include <OptiXRenderer/Shading/ShadingModels/DefaultShading.h>
#include <OptiXRenderer/RNG.h>

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Core/Array.h>
#include <Bifrost/Math/Utils.h>

#include <StbImageWriter/StbImageWriter.h>

#include <fstream>

using namespace Bifrost;
using namespace Bifrost::Assets;
using namespace optix;
using namespace OptiXRenderer;
using namespace OptiXRenderer::Shading::BSDFs;
using namespace OptiXRenderer::Shading::ShadingModels;

typedef BSDFSample(*SampleRoughBSDF)(float3 tint, float roughness, float3 wo, float2 random_sample);

double estimate_rho(float3 wo, float roughness, unsigned int sample_count, SampleRoughBSDF sample_rough_BSDF) {

    const float3 tint = make_float3(1.0f, 1.0f, 1.0f);

    Core::Array<double> throughput = Core::Array<double>(sample_count);
    for (unsigned int s = 0; s < sample_count; ++s) {
        float2 rng_sample = RNG::sample02(s);
        BSDFSample sample = sample_rough_BSDF(tint, roughness, wo, rng_sample);
        if (is_PDF_valid(sample.PDF))
            throughput[s] = sample.reflectance.x * sample.direction.z / sample.PDF;
        else
            throughput[s] = 0.0;
    }

    return Math::sort_and_pairwise_summation(throughput.begin(), throughput.end()) / sample_count;
}

Image estimate_rho(unsigned int width, unsigned int height, unsigned int sample_count, SampleRoughBSDF sample_rough_BSDF) {
    Image rho_image = Images::create2D("rho", PixelFormat::RGB_Float, 1.0f, Math::Vector2ui(width, height));
    Math::RGB* rho_image_pixels = rho_image.get_pixels<Math::RGB>();

    #pragma omp parallel for
    for (int y = 0; y < int(height); ++y) {
        float roughness = y / float(height - 1);
        for (int x = 0; x < int(width); ++x) {
            float cos_theta = max(0.000001f, x / float(width - 1));
            float3 wo = make_float3(sqrt(1.0f - cos_theta * cos_theta), 0.0f, cos_theta);
            double rho = estimate_rho(wo, roughness, sample_count, sample_rough_BSDF);
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

std::string get_rho_type(int dimensions) {
    std::ostringstream out;
    if (dimensions == 1)
        out << "float";
    else
        out << "Vector" << dimensions << "f";
    return out.str();
}

template <int ElementDimensions>
void output_brdf(Image image, int sample_count, const std::string& filename, const std::string& data_name, const std::string& description) {

    unsigned int width = image.get_width();
    unsigned int height = image.get_height();
    Math::RGB* image_pixels = image.get_pixels<Math::RGB>();

    std::string rho_type = get_rho_type(ElementDimensions);

    std::ofstream out_header(filename);
    out_header <<
        "// " << description << "\n"
        "// ------------------------------------------------------------------------------------------------\n"
        "// Copyright (C) 2018, Bifrost. See AUTHORS.txt for authors\n"
        "//\n"
        "// This program is open source and distributed under the New BSD License.\n"
        "// See LICENSE.txt for more detail.\n"
        "// ------------------------------------------------------------------------------------------------\n"
        "// Generated by MaterialPrecomputations application.\n"
        "// ------------------------------------------------------------------------------------------------\n"
        "\n"
        "#include <Bifrost/Assets/Shading/Fittings.h>\n"
        "#include <Bifrost/Math/Utils.h>\n"
        "\n";
    if (ElementDimensions > 1)
        out_header << "using Bifrost::Math::" << rho_type << ";\n\n";
    out_header <<
        "namespace Bifrost::Assets::Shading::Rho {\n"
        "\n";
    out_header << "const int " << data_name << "_sample_count = " << sample_count << "u;\n"
        "const int " << data_name << "_angle_sample_count = " << width << "u;\n"
        "const int " << data_name << "_roughness_sample_count = " << height << "u;\n"
        "\n"
        "const " << rho_type << " " << data_name << "[] = {\n";

    for (int y = 0; y < int(height); ++y) {
        float roughness = y / float(height - 1);
        out_header << "    // Roughness " << roughness << "\n";
        out_header << "    ";
        for (int x = 0; x < int(width); ++x) {
            Math::RGB& rho = image_pixels[x + y * width];
            if (ElementDimensions == 1)
                out_header << format_float(rho.r) << ", ";
            else if (ElementDimensions == 2)
                out_header << "Vector2f(" << format_float(rho.r) << ", " << format_float(rho.g) << "), ";
            else if (ElementDimensions == 3)
                out_header << "Vector3f(" << format_float(rho.r) << ", " << format_float(rho.g) << ", " << format_float(rho.b) << "), ";
        }
        out_header << "\n";
    }

    out_header <<
        "};\n"
        "\n"
        "" << rho_type << " sample_" << data_name << "(float wo_dot_normal, float roughness) {\n"
        "    using namespace Bifrost::Math;\n"
        "\n"
        "    float roughness_coord = roughness * (" << data_name << "_roughness_sample_count - 1);\n"
        "    int lower_roughness_row = int(roughness_coord);\n"
        "    int upper_roughness_row = min(lower_roughness_row + 1, " << data_name << "_roughness_sample_count - 1);\n"
        "\n"
        "    float wo_dot_normal_coord = wo_dot_normal * (" << data_name << "_angle_sample_count - 1);\n"
        "    int lower_wo_dot_normal_column = int(wo_dot_normal_coord);\n"
        "    int upper_wo_dot_normal_column = min(lower_wo_dot_normal_column + 1, " << data_name << "_angle_sample_count - 1);\n"
        "\n"
        "    // Interpolate by wo_dot_normal\n"
        "    float wo_dot_normal_t = wo_dot_normal * (" << data_name << "_angle_sample_count - 1) - lower_wo_dot_normal_column;\n"
        "    const " << rho_type << "* lower_rho_row = " << data_name << " + lower_roughness_row * " << data_name << "_roughness_sample_count;\n"
        "    " << rho_type << " lower_rho = lerp(lower_rho_row[lower_wo_dot_normal_column], lower_rho_row[upper_wo_dot_normal_column], wo_dot_normal_t);\n"
        "\n"
        "    const " << rho_type << "* upper_rho_row = " << data_name << " + upper_roughness_row * " << data_name << "_roughness_sample_count;\n"
        "    " << rho_type << " upper_rho = lerp(upper_rho_row[lower_wo_dot_normal_column], upper_rho_row[upper_wo_dot_normal_column], wo_dot_normal_t);\n"
        "\n"
        "    // Interpolate by roughness\n"
        "    float roughness_t = roughness_coord - lower_roughness_row;\n"
        "    return lerp(lower_rho, upper_rho, roughness_t);\n"
        "}\n"
        "\n"
        "} // NS Bifrost::Assets::Shading::Rho\n";

    out_header.close();
}

// Given a max PDF and cos(theta) compute the corresponding alpha of the GGX distribution with that max PDF.
void estimate_alpha_from_max_PDF(int cos_theta_count, int max_PDF_count, const std::string& filename) {
    const int sample_count = max_PDF_count * cos_theta_count;
    constexpr float k = 1.0f; // Found to give a decent distribution of alphas, where decent is defined as the distribution of neighbouring alphas in the lookup table with the lowest standard deviatino
    auto encode_PDF = [=](float pdf) -> float {
        float non_linear_PDF = pdf / (k + pdf);
        return (non_linear_PDF - 0.13f) / 0.87f;
    };

    // Cost function for determining k
    // auto std_dev_error = [=](float alphas[]) -> float {
    //     Math::Statistics<double> stats;
    //     for (int c = 0; c < cos_theta_count - 1; ++c) {
    //         for (int x = 0; x < pdf_count - 1; ++x) {
    //             float alpha = alphas[x + c * pdf_count];
    //             double vertical_diff = alpha - alphas[x + 1 + c * pdf_count];
    //             stats.add(vertical_diff);
    //             double horizontal_diff = alpha - alphas[x + (c + 1) * pdf_count];
    //             stats.add(horizontal_diff);
    //         }
    //     }
    //     return float(stats.standard_deviation());
    // };

    struct PDFSample {
        float alpha;
        float encoded_PDF;
    };

    // Fill the alpha lookup table
    float* alphas = new float[sample_count];
    for (int c = 0; c < cos_theta_count; ++c) {
        float cos_theta = fmaxf(c / (cos_theta_count - 1.0f), 0.0001f);
        float3 wo = { sqrt(1 - pow2(cos_theta)), 0.0f, cos_theta };
        float3 wi = { -wo.x, -wo.y, wo.z };

        for (int t = 0; t < max_PDF_count; ++t) {
            int index = t + c * max_PDF_count;
            float encoded_target_PDF = t / (max_PDF_count - 1.0f);

            // Binary search to find the alpha that hits the target PDF
            float prev_alpha = t == 0 ? 1.0f : alphas[index - 1];
            float3 reflected_wi = { -wo.x, -wo.y, wo.z };
            PDFSample low_PDF_sample = { prev_alpha, encode_PDF(GGX_R::PDF(prev_alpha, wo, reflected_wi)) };
            PDFSample high_PDF_sample = { 0.0f, encode_PDF(GGX_R::PDF(0.00000000001f, wo, reflected_wi)) };

            float alpha = 0.0f;
            if (encoded_target_PDF >= high_PDF_sample.encoded_PDF)
                alpha = high_PDF_sample.alpha;
            else if (encoded_target_PDF <= low_PDF_sample.encoded_PDF)
                alpha = low_PDF_sample.alpha;
            else {
                PDFSample middle_sample;
                do {
                    float middle_alpha = (low_PDF_sample.alpha + high_PDF_sample.alpha) * 0.5f;
                    middle_sample = { middle_alpha, encode_PDF(GGX_R::PDF(middle_alpha, wo, reflected_wi)) };
                    if (encoded_target_PDF < middle_sample.encoded_PDF)
                        high_PDF_sample = middle_sample;
                    else
                        low_PDF_sample = middle_sample;
                } while (abs(middle_sample.encoded_PDF - encoded_target_PDF) / (middle_sample.encoded_PDF + encoded_target_PDF) > 0.000001f);

                alpha = middle_sample.alpha;
            }

            alphas[index] = alpha;
        }
    }

    // Assert that the alpha map span the full range of [0, 1] per angle.
    // If not then the encoding may be clipping information.
    for (int y = 0; y < cos_theta_count; ++y) {
        float first_column_alpha = alphas[0 + y * max_PDF_count];
        if (first_column_alpha != 1.0f)
            throw std::exception("Alpha in first column must be 1");
        float last_column_alpha = alphas[(max_PDF_count - 1) + y * max_PDF_count];
        if (last_column_alpha != 0.0f)
            throw std::exception("Alpha in first column must be 0");
    }

    { // Output the fitting
        std::ofstream out_header(filename);
        out_header <<
            "// Estimate the alpha of the GGX bounded VNDF distribution based on the maximal PDF when sampling the bounded VNDF\n"
            "// and the angle between the view direction and the normal.\n"
            "// ------------------------------------------------------------------------------------------------\n"
            "// Copyright (C) 2018, Bifrost. See AUTHORS.txt for authors\n"
            "//\n"
            "// This program is open source and distributed under the New BSD License.\n"
            "// See LICENSE.txt for more detail.\n"
            "// ------------------------------------------------------------------------------------------------\n"
            "// Generated by MaterialPrecomputations application.\n"
            "// ------------------------------------------------------------------------------------------------\n"
            "\n"
            "#include <Bifrost/Assets/Shading/Fittings.h>\n"
            "#include <Bifrost/Math/Utils.h>\n"
            "\n"
            "namespace Bifrost::Assets::Shading::Estimate_GGX_bounded_VNDF_alpha {\n"
            "\n"
            "const int alpha_sample_count = " << sample_count << ";\n"
            "const int wo_dot_normal_sample_count = " << cos_theta_count << ";\n"
            "const int max_PDF_sample_count = " << max_PDF_count << ";\n"
            "\n"
            "const float alphas[] = {\n";

        for (int y = 0; y < cos_theta_count; ++y) {
            float cos_theta = y / float(cos_theta_count - 1u);
            out_header << "    // wo_dot_normal " << cos_theta << "\n";
            out_header << "    ";
            for (int x = 0; x < max_PDF_count; ++x)
                out_header << alphas[x + y * max_PDF_count] << ", ";
            out_header << "\n";
        }

        out_header <<
            "};\n"
            "\n"
            "float encode_PDF(float pdf) {\n"
            "    float non_linear_PDF = pdf / (1.0f + pdf);\n"
            "    return (non_linear_PDF - 0.13f) / 0.87f;\n"
            "}\n"
            "float decode_PDF(float encoded_pdf) {\n"
            "    float non_linear_PDF = encoded_pdf * 0.87f + 0.13f;\n"
            "    return non_linear_PDF / (1.0f - non_linear_PDF);\n"
            "}\n"
            "\n"
            "float estimate_alpha(float wo_dot_normal, float max_PDF) {\n"
            "    using namespace Bifrost::Math;\n"
            "\n"
            "    float encoded_PDF = encode_PDF(max_PDF);\n"
            "\n"
            "    float wo_dot_normal_coord = wo_dot_normal * (wo_dot_normal_sample_count - 1);\n"
            "    int lower_wo_dot_normal_row = int(wo_dot_normal_coord);\n"
            "    int upper_wo_dot_normal_row = min(lower_wo_dot_normal_row + 1, wo_dot_normal_sample_count - 1);\n"
            "\n"
            "    float encoded_PDF_coord = encoded_PDF * (max_PDF_sample_count - 1);\n"
            "    int lower_encoded_PDF_column = int(encoded_PDF_coord);\n"
            "    int upper_encoded_PDF_column = min(lower_encoded_PDF_column + 1, max_PDF_sample_count - 1);\n"
            "\n"
            "    // Interpolate by encoded PDF\n"
            "    float encoded_PDF_t = encoded_PDF_coord - lower_encoded_PDF_column;\n"
            "    const float* lower_alpha_row = alphas + lower_wo_dot_normal_row * wo_dot_normal_sample_count;\n"
            "    float lower_alpha = lerp(lower_alpha_row[lower_encoded_PDF_column], lower_alpha_row[upper_encoded_PDF_column], encoded_PDF_t);\n"
            "\n"
            "    const float* upper_alpha_row = alphas + upper_wo_dot_normal_row * wo_dot_normal_sample_count;\n"
            "    float upper_alpha = lerp(upper_alpha_row[lower_encoded_PDF_column], upper_alpha_row[upper_encoded_PDF_column], encoded_PDF_t);\n"
            "\n"
            "    // Interpolate by wo_dot_normal\n"
            "    float wo_dot_normal_t = wo_dot_normal_coord - lower_wo_dot_normal_row;\n"
            "    return lerp(lower_alpha, upper_alpha, wo_dot_normal_t);\n"
            "}\n"
            "\n"
            "} // NS Bifrost::Assets::Shading::Estimate_GGX_bounded_VNDF_alpha\n";

        out_header.close();
    }

    delete[] alphas;
}

int main(int argc, char** argv) {
    printf("Material Precomputations\n");

    std::string output_dir = argc >= 2? argv[1] : std::string(BIFROST_SHADING_DIR);
    printf("output_dir: %s\n", output_dir.c_str());

    const unsigned int width = 64, height = 64, sample_count = 4096;

    Images::allocate(1);

    fit_GGX_rho_approximation(output_dir);

    // Given a max PDF and cos(theta) compute the corresponding alpha of the GGX distribution with that max PDF.
    estimate_alpha_from_max_PDF(32, 32, output_dir + "EstimateGGXBoundedVNDFAlpha.cpp");

    { // Default shading albedo.

        // Compute the directional-hemispherical reflectance function, albedo, by monte carlo integration and store the result in a texture and as an array in a header file.
        // The diffuse and specular components are separated by tinting the diffuse layer with red and keeping the specular layer white.
        // The albedo is computed via monte carlo integration by assuming that the material is lit by a uniform infinitely far away area light with an intensity of one.
        // As the base material is green it has no contribution to the red and blue channels, which means that these contain the albedo of the specular component.
        // The green channel contains the contribution of both the specular and diffuse components and the diffuse contribution alone can be found by subtracting the specular contribution from the green channel.
        // Notes
        // * Fresnel base reflectivity is set to zero. This is completely unrealistic, but gives us the largest possible range between full diffuse and full specular.

        // Specular material.
        OptiXRenderer::Material material_params = {};
        material_params.tint = optix::make_float3(1.0f, 0.0f, 0.0f);
        material_params.metallic = 0.0f;
        material_params.specularity = 0.0f;

        Image rho = Images::create2D("rho", PixelFormat::RGB_Float, 1.0f, Math::Vector2ui(width, height));
        Math::RGB* rho_pixels = rho.get_pixels<Math::RGB>();

        for (int y = 0; y < int(height); ++y) {
            material_params.roughness = y / float(height - 1u);
            #pragma omp parallel for
            for (int x = 0; x < int(width); ++x) {

                float cos_theta = max(0.000001f, x / float(width - 1));
                float3 wo = make_float3(sqrt(1.0f - cos_theta * cos_theta), 0.0f, cos_theta);

                DefaultShading material = DefaultShading(material_params, wo.z);

                Core::Array<double> specular_throughput = Core::Array<double>(sample_count);
                Core::Array<double> total_throughput = Core::Array<double>(sample_count);
                for (unsigned int s = 0; s < sample_count; ++s) {

                    float3 rng_sample = make_float3(RNG::sample02(s), (s + 0.5f) / sample_count);
                    BSDFSample sample = material.sample(wo, rng_sample);
                    if (is_PDF_valid(sample.PDF)) {
                        total_throughput[s] = sample.reflectance.x * sample.direction.z / sample.PDF;
                        specular_throughput[s] = sample.reflectance.y * sample.direction.z / sample.PDF;
                    } else
                        total_throughput[s] = specular_throughput[s] = 0.0;
                }

                double specular_rho = Math::sort_and_pairwise_summation(specular_throughput.begin(), specular_throughput.end()) / sample_count;
                double total_rho = Math::sort_and_pairwise_summation(total_throughput.begin(), total_throughput.end()) / sample_count;
                double diffuse_rho = total_rho - specular_rho;
                rho_pixels[x + y * width] = Math::RGB(float(diffuse_rho), float(specular_rho), 0.0f);
            }
        }

        // Store.
        StbImageWriter::write(rho, output_dir + "DefaultShadingRho.png");
        output_brdf<2>(rho, sample_count, output_dir + "DefaultShadingRho.cpp", "default_shading",
            "Directional-hemispherical reflectance for default shaded material.");
    }

    { // Compute Burley rho.

        Image rho = estimate_rho(width, height, sample_count, Burley::sample);

        // Store.
        StbImageWriter::write(rho, output_dir + "BurleyRho.png");
        output_brdf<1>(rho, sample_count, output_dir + "BurleyRho.cpp", "burley", "Directional-hemispherical reflectance for Burley.");
    }

    { // Compute OrenNayar rho.

        Image rho = estimate_rho(width, height, sample_count, OrenNayar::sample);

        // Store.
        StbImageWriter::write(rho, output_dir + "OrenNayarRho.png");
        output_brdf<1>(rho, sample_count, output_dir + "OrenNayarRho.cpp", "oren_nayar", "Directional-hemispherical reflectance for OrenNayar.");
    }

    { // Compute GGX rho.

        static auto sample_ggx = [](float3 tint, float roughness, float3 wo, float2 random_sample) -> BSDFSample {
            float alpha = GGX::alpha_from_roughness(roughness);
            return GGX_R::sample(alpha, 1, wo, random_sample);
        };

        Image rho = estimate_rho(width, height, sample_count, sample_ggx);

        // Store.
        StbImageWriter::write(rho, output_dir + "GGXRho.png");
        output_brdf<1>(rho, sample_count, output_dir + "GGXRho.cpp", "GGX", "Directional-hemispherical reflectance for GGX.");
    }

    { // Compute GGX with fresnel rho.

        static auto sample_ggx_with_fresnel = [](float3 tint, float roughness, float3 wo, float2 random_sample) -> BSDFSample {
            float alpha = GGX::alpha_from_roughness(roughness);
            return GGX_R::sample(alpha, 0, wo, random_sample);
        };

        Image rho = estimate_rho(width, height, sample_count, sample_ggx_with_fresnel);

        // Store.
        StbImageWriter::write(rho, output_dir + "GGXWithFresnelRho.png");
        output_brdf<1>(rho, sample_count, output_dir + "GGXWithFresnelRho.cpp", "GGX_with_fresnel",
            "Directional-hemispherical reflectance for GGX with fresnel factor.");
    }

    return 0;
}
