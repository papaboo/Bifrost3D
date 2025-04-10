// Fit map from maximal possible PDF to GGX reflection alpha.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _GGX_ALPHA_FROM_MAX_PDF_H_
#define _GGX_ALPHA_FROM_MAX_PDF_H_

#include <OptiXRenderer/Shading/BSDFs/GGX.h>

#include <fstream>

// Given a max PDF and cos(theta) compute the corresponding alpha of the GGX distribution with that max PDF.
void estimate_alpha_from_max_PDF(int cos_theta_count, int max_PDF_count, const std::string& filename) {
    using namespace optix;
    using namespace OptiXRenderer;
    using namespace OptiXRenderer::Shading::BSDFs;

    const int sample_count = max_PDF_count * cos_theta_count;
    constexpr float k = 1.0f; // Found to give a decent distribution of alphas, where decent is defined as the distribution of neighbouring alphas in the lookup table with the lowest standard deviatino
    auto encode_PDF = [=](PDF pdf) -> float {
        if (pdf.is_delta_dirac())
            return 1;

        float non_linear_PDF = pdf.value() / (k + pdf.value());
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
        float3 wo = { sqrt(1 - OptiXRenderer::pow2(cos_theta)), 0.0f, cos_theta };
        float3 wi = { -wo.x, -wo.y, wo.z };

        for (int t = 0; t < max_PDF_count; ++t) {
            int index = t + c * max_PDF_count;
            float encoded_target_PDF = t / (max_PDF_count - 1.0f);

            // Binary search to find the alpha that hits the target PDF
            float prev_alpha = t == 0 ? 1.0f : alphas[index - 1];
            float3 reflected_wi = { -wo.x, -wo.y, wo.z };
            PDFSample low_PDF_sample = { prev_alpha, encode_PDF(GGX_R::pdf(prev_alpha, wo, reflected_wi)) };
            PDFSample high_PDF_sample = { 0.0f, encode_PDF(GGX_R::pdf(0.00000000001f, wo, reflected_wi)) };

            float alpha = 0.0f;
            if (encoded_target_PDF >= high_PDF_sample.encoded_PDF)
                alpha = high_PDF_sample.alpha;
            else if (encoded_target_PDF <= low_PDF_sample.encoded_PDF)
                alpha = low_PDF_sample.alpha;
            else {
                PDFSample middle_sample;
                do {
                    float middle_alpha = (low_PDF_sample.alpha + high_PDF_sample.alpha) * 0.5f;
                    middle_sample = { middle_alpha, encode_PDF(GGX_R::pdf(middle_alpha, wo, reflected_wi)) };
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
            "// Copyright (C) Bifrost. See AUTHORS.txt for authors\n"
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
            "    float encoded_PDF = (non_linear_PDF - 0.13f) / 0.87f;\n"
            "    if (isnan(encoded_PDF))\n"
            "        encoded_PDF = 1.0f;\n"
            "    return encoded_PDF;\n"
            "}\n"
            "\n"
            "float estimate_alpha(float wo_dot_normal, float max_PDF) {\n"
            "    using namespace Bifrost::Math;\n"
            "\n"
            "    float encoded_PDF = encode_PDF(max_PDF);\n"
            "    encoded_PDF = fminf(1.0f, encoded_PDF);\n"
            "    encoded_PDF = fmaxf(0.0f, encoded_PDF);\n"
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

#endif // _GGX_ALPHA_FROM_MAX_PDF_H_