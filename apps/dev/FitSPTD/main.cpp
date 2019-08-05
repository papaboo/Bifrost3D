// Spherical pivot transformation distribution fitting of BRDFs.
// ------------------------------------------------------------------------------------------------
// Thanks to Eric Heitz and Jonathan Dupuy for the original code.
// ------------------------------------------------------------------------------------------------

#include "Brdf.h"
#include "SPTD.h"

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Math/NelderMead.h>
#include <StbImageWriter/StbImageWriter.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace SPTD;

// number of samples used to compute the error during fitting
const int sample_count = 32;

struct DirectionAndRho {
    float3 direction;
    float rho;
};

// compute average vector of the BRDF
template <typename BRDF>
DirectionAndRho compute_average_sample(BRDF brdf, float3 wo, float alpha) {
    float3 summed_direction = make_float3(0.0f);
    float norm = 0.0;

    for (int j = 0; j < sample_count; ++j)
        for (int i = 0; i < sample_count; ++i) {

            const float U1 = (i + 0.5f) / (float)sample_count;
            const float U2 = (j + 0.5f) / (float)sample_count;

            // sample
            auto brdf_sample = brdf.sample(wo, alpha, U1, U2);
            if (is_PDF_valid(brdf_sample.PDF)) {
                // accumulate
                float weight = (brdf_sample.PDF > 0) ? brdf_sample.weight.x / brdf_sample.PDF : 0.0f;
                norm += weight;
                summed_direction += weight * brdf_sample.direction;
            }
        }

    return { normalize(summed_direction), norm / float(sample_count * sample_count) };
}

// compute the error between the BRDF and the pivot using Multiple Importance Sampling
template <typename BRDF>
float compute_error(Pivot pivot, BRDF brdf, float3 wo, float alpha) {
    double error = 0.0;
    int valid_sample_count = 0;

    for (int j = 0; j < sample_count; ++j)
        for (int i = 0; i < sample_count; ++i) {

            const float U1 = (i + 0.5f) / (float)sample_count;
            const float U2 = (j + 0.5f) / (float)sample_count;

            /*
            // error with MIS weight
            auto error_from_wi = [&](float3 wi) -> double {
                auto brdf_response = brdf.eval(wo, wi, alpha);
                float eval_pivot = pivot.eval(wi);
                float pdf_pivot = eval_pivot / pivot.amplitude;
                double error = brdf_response.weight.x - eval_pivot;
                return error * error / (pdf_pivot + brdf_response.PDF);
            };

            { // importance sample LTC
                const float3 wi = pivot.sample(U1, U2);
                if (wi.z >= 0.0f) {
                    error += error_from_wi(wi);
                    ++valid_sample_count;
                }
            }

            { // importance sample BRDF
                const float3 wi = brdf.sample_direction(wo, alpha, U1, U2);
                if (wi.z >= 0.0f) {
                    error += error_from_wi(wi);
                    ++valid_sample_count;
                }
            }
            */

            // error with MIS weight
            auto compute_error = [&](float3 sampled_wi, BSDFResponse brdf_response) -> double {
                float eval_pivot = pivot.eval(sampled_wi);
                float pdf_pivot = eval_pivot / pivot.amplitude;
                double error = brdf_response.weight.x - eval_pivot;
                return error * error / (pdf_pivot + brdf_response.PDF);
            };

            { // importance sample LTC
                const float3 wi = pivot.sample(U1, U2);
                if (wi.z >= 0.0f) {
                    auto brdf_response = brdf.eval(wo, wi, alpha);
                    error += compute_error(wi, brdf_response);
                    ++valid_sample_count;
                }
            }

            { // importance sample BRDF
                auto brdf_sample = brdf.sample(wo, alpha, U1, U2);
                if (is_PDF_valid(brdf_sample.PDF)) {
                    BSDFResponse brdf_response = { brdf_sample.weight, brdf_sample.PDF };
                    error += compute_error(brdf_sample.direction, brdf_response);
                    ++valid_sample_count;
                }
            }
        }

    return float(error / valid_sample_count);
}

template <typename BRDF>
struct PivotFitter {

    PivotFitter(Pivot& pivot, BRDF brdf, float3 wo, float alpha)
        : pivot(pivot), brdf(brdf), wo(wo), alpha(alpha) { }

    void update(float* params) {
        pivot.distance = clamp(params[0], 0.001f, 0.999f);
        pivot.theta = clamp(params[1], -1.5707f, 0.0f);

        // Mirror around the borders of the domain.
        float distance_diff = pivot.distance - params[0];
        pivot.distance = params[0] += 1.1f * distance_diff;
        float theta_diff = pivot.theta - params[1];
        pivot.theta = params[1] += 1.1f * theta_diff;
    }

    float operator()(float* params) {
        update(params);

        // NOTE Ideally we'd assign a higher weight to errors where the distance and theta are outside the valid domain, 
        // but in practice that yields less diserable results than simply mirroring vertices around the domain border.
        // float distance_offset = params[0] - pivot.distance;
        // float theta_offset = params[1] - pivot.theta;
        // float error_scale = 1.0f + 32.0f * (distance_offset * distance_offset + theta_offset * theta_offset);

        return compute_error(pivot, brdf, wo, alpha);
    }

    BRDF brdf;
    Pivot& pivot;

    const float3& wo;
    float alpha;
};

// Fit using nelder-mead.
template <typename BRDF>
void fit(Pivot& pivot, BRDF brdf, float3 wo, float alpha, float epsilon = 0.05f) {

    float start_fit[2] = { pivot.distance, pivot.theta };
    float result_fit[2];

    PivotFitter<BRDF> fitter(pivot, brdf, wo, alpha);
    float error = Bifrost::Math::nelder_mead<2>(result_fit, start_fit, epsilon, 1e-5f, 200, fitter);

    // Update pivot with best fitting values
    fitter.update(result_fit);
}

// fit data
template <typename BRDF>
void fit_pivot(Pivot* pivots, const int size, BRDF brdf) {

    // Loop over theta and alpha.
    #pragma omp parallel for
    for (int i = 0; i < size * size; ++i) {
        int t = i % size, r = i / size;

        float cos_theta = t / float(size - 1);
        float theta = fminf(1.57f, acosf(cos_theta));
        // std::cout << "cos_theta: " << cos_theta << ", theta: " << theta << ", cos(theta): " << cosf(theta) << std::endl;
        const float3 wo = make_float3(sinf(theta), 0, cosf(theta));

        float roughness = r / float(size - 1);
        float alpha = fmaxf(roughness * roughness, 0.001f); // OptiXRenderer::Shading::BSDFs::GGX::alpha_from_roughness(roughness); // TODO The minimal alpha should be reduced, but that leads to bad fits on nearly specular surfaces.
        auto average_sample = compute_average_sample(brdf, wo, alpha);

        // init
        Pivot& pivot = pivots[t + r * size];
        pivot.distance = 0.5f + 0.49f * (1.0f - alpha);
        pivot.theta = acos(average_sample.direction.z) * sign(average_sample.direction.x);
        pivot.amplitude = average_sample.rho;

        // Fit
        float epsilon = 0.05f;
        fit(pivot, brdf, wo, alpha, epsilon);

        // std::cout << "  [cos_theta: " << V.z << ", roughness: " << roughness << "]: Pivot: [theta: " << pivot.theta << ", distance: " << pivot.distance << ", rho: " << pivot.amplitude << "]" << std::endl;
    }
}

void output_fit_header(Pivot* pivots, unsigned int width, unsigned int height, const std::string& filename, const std::string& data_name, const std::string& description) {

    using namespace std;

    auto format_float = [](float v) -> string {
        ostringstream out;
        out << v;
        if (out.str().length() == 1)
            out << ".0f";
        else
            out << "f";
        return out.str();
    };

    ofstream out_header(filename);
    out_header <<
        "// " << description << "\n"
        "// ------------------------------------------------------------------------------------------------\n"
        "// Copyright (C) 2018, Bifrost. See AUTHORS.txt for authors\n"
        "//\n"
        "// This program is open source and distributed under the New BSD License.\n"
        "// See LICENSE.txt for more detail.\n"
        "// ------------------------------------------------------------------------------------------------\n"
        "// Generated by FitSPTD application.\n"
        "// ------------------------------------------------------------------------------------------------\n"
        "\n"
        "#include <Bifrost/Assets/Shading/Fittings.h>\n"
        "\n"
        "using Bifrost::Math::Vector3f;\n"
        "\n"
        "namespace Bifrost {\n"
        "namespace Assets {\n"
        "namespace Shading {\n"
        "\n";
    out_header << "const unsigned int " << data_name << "_SPTD_fit_angular_sample_count = " << width << "u;\n"
        "const unsigned int " << data_name << "_SPTD_fit_roughness_sample_count = " << height << "u;\n"
        "\n"
        "const Vector3f " << data_name << "_SPTD_fit[] = {\n";

    for (int y = 0; y < int(height); ++y) {
        float roughness = y / float(height - 1u);
        out_header << "    // Roughness " << roughness << "\n";
        out_header << "    ";
        for (int x = 0; x < int(width); ++x) {
            Pivot& fit = pivots[x + y * width];
            out_header << "Vector3f(" << format_float(fit.distance) << ", " << format_float(fit.theta) << ", " << format_float(fit.amplitude) << "), ";
        }
        out_header << "\n";
    }

    out_header <<
        "};\n"
        "\n"
        "Vector3f " << data_name << "_SPTD_fit_lookup(float cos_theta, float roughness) {\n"
        "    float u = roughness;\n"
        "    int ui = int(u * " << data_name << "_SPTD_fit_roughness_sample_count);\n"
        "    float v = 2.0f * acosf(cos_theta) / 3.14159f;\n"
        "    int vi = int(v * " << data_name << "_SPTD_fit_angular_sample_count);\n"
        "    return " << data_name << "_SPTD_fit[vi + ui * " << data_name << "_SPTD_fit_angular_sample_count];\n"
        "}\n"
        "\n"
        "} // NS Shading\n"
        "} // NS Assets\n"
        "} // NS Bifrost\n";

    out_header.close();
}

void output_fit_image(Pivot* pivots, unsigned int width, unsigned int height, const std::string& filename) {
    using namespace Bifrost::Assets;
    using namespace Bifrost::Math;

    Images::UID image_ID = Images::create2D("pivot", PixelFormat::RGB_Float, 1.0f, Vector2ui(width, height));
    RGB* pixels = Images::get_pixels<RGB>(image_ID);
    for (unsigned int i = 0; i < width * height; ++i) {
        Pivot& pivot = pivots[i];
        pixels[i] = RGB(pivot.distance, pivot.theta / -PIf * 2.0f, pivot.amplitude);
    }

    StbImageWriter::write(image_ID, filename);

    Images::destroy(image_ID);
}

template <typename BRDF>
void output_error(const Pivot* const pivots, BRDF brdf, int width, int height, const std::string& distribution_name) {
    double error = 0.0;
    for (int i = 0; i < width * height; ++i) {
        int t = i % width, a = i / width;

        float cos_theta = t / float(height - 1);
        float theta = fminf(1.57f, acosf(cos_theta));
        const float3 wo = make_float3(sinf(theta), 0, cosf(theta));

        float roughness = a / float(width - 1);
        float alpha = fmaxf(roughness * roughness, 0.001f); // OptiXRenderer::Shading::BSDFs::GGX::alpha_from_roughness(roughness); // TODO The minimal alpha should be reduced, but that leads to bad fits on nearly specular surfaces.

        const Pivot& pivot = pivots[t + a * width];

        error += compute_error(pivot, brdf, wo, alpha);
    }

    std::cout << distribution_name << " error: " << error / (width * height) << std::endl;
}

template <typename BRDF>
void output_gradient_error(const Pivot* const pivots, BRDF brdf, int width, int height, const std::string& distribution_name) {

    float3 mins = { 1e30f, 1e30f, 1e30f };
    float3 maxs = { -1e30f, -1e30f, -1e30f };
    for (int i = 0; i < width * height; ++i) {
        const Pivot& pivot = pivots[i];
        mins = { fminf(mins.x, pivot.distance), fminf(mins.y, pivot.theta), fminf(mins.z, cosf(pivot.theta)) };
        maxs = { fmaxf(maxs.x, pivot.distance), fmaxf(maxs.y, pivot.theta), fmaxf(maxs.z, cosf(pivot.theta)) };
    }
    float3 range = maxs - mins;

    double3 error = { 0.0, 0.0, 0.0 };
    for (int i = 0; i < width * height; ++i) {
        int x = i % width, y = i / width;
        const Pivot& pivot = pivots[x + y * width];

        // Computes the normalized root mean square error.
        auto add_pivot_error = [&](Pivot prev_pivot) {
            error.x += pow2((pivot.distance - prev_pivot.distance) / range.x);
            error.y += pow2((pivot.theta - prev_pivot.theta) / range.y);
            error.z += pow2((cosf(pivot.theta) - cosf(prev_pivot.theta)) / range.z);
        };

        if (x > 0)
            add_pivot_error(pivots[x - 1 + y * width]);

        if (y > 0)
            add_pivot_error(pivots[x + (y - 1) * width]);
    }

    std::cout << distribution_name << " gradient error: distance: " << error.x / (width * height) << ", theta: " << error.y / (width * height) << ", cos_theta: " << error.z / (width * height) << std::endl;
}

int main(int argc, char* argv[]) {
    printf("SPTD fitting\n");

    std::string output_dir = argc >= 2 ? argv[1] : std::string(BIFROST_SHADING_DIR);
    printf("output_dir: %s\n", output_dir.c_str());

    Bifrost::Assets::Images::allocate(2);

    // size of precomputed table (theta, roughness)
    const int size = 64;

    // allocate data
    Pivot* pivots = new Pivot[size * size];

    { // GGX
        BRDF::GGX brdf;
        fit_pivot(pivots, size, brdf);

        output_fit_header(pivots, size, size, output_dir + "GGXSPTDFit.cpp", "GGX", "GGX fit for spherical pivot transformed distributions.");
        output_fit_image(pivots, size, size, output_dir + "GGXSPTDFit.png");

        output_error(pivots, brdf, size, size, "GGX");
        output_gradient_error(pivots, brdf, size, size, "GGX");
    }

    // delete data
    delete[] pivots;

    return 0;
}