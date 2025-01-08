// Test OptiXRenderer's OrenNayar BRDF.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _OPTIXRENDERER_BSDFS_OREN_NAYAR_TEST_H_
#define _OPTIXRENDERER_BSDFS_OREN_NAYAR_TEST_H_

#include <BSDFTestUtils.h>

#include <OptiXRenderer/Shading/BSDFs/OrenNayar.h>
#include <OptiXRenderer/Shading/ShadingModels/DefaultShading.h>

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Math/Line.h>
#include <StbImageWriter/StbImageWriter.h>

#include <gtest/gtest.h>

using namespace Bifrost::Assets;
using namespace Bifrost::Math;

namespace OptiXRenderer {

class OrenNayarWrapper {
public:
    float m_roughness;
    optix::float3 m_tint;

    OrenNayarWrapper(float roughness, optix::float3 tint = { 1, 1, 1})
        : m_roughness(roughness), m_tint(tint) {}

    optix::float3 evaluate(optix::float3 wo, optix::float3 wi) const {
        return Shading::BSDFs::OrenNayar::evaluate(m_tint, m_roughness, wo, wi, true);
    }

    float PDF(optix::float3 wo, optix::float3 wi) const {
        return Shading::BSDFs::OrenNayar::PDF(m_roughness, wo, wi);
    }

    BSDFResponse evaluate_with_PDF(optix::float3 wo, optix::float3 wi) const {
        return Shading::BSDFs::OrenNayar::evaluate_with_PDF(m_tint, m_roughness, wo, wi, true);
    }

    BSDFSample sample(optix::float3 wo, optix::float3 random_sample) const {
        return Shading::BSDFs::OrenNayar::sample(m_tint, m_roughness, wo, optix::make_float2(random_sample), true);
    }

    std::string to_string() const {
        std::ostringstream out;
        out << "OrenNayar: roughness: " << m_roughness << ", tint: [" << m_tint.x << ", " << m_tint.y << ", " << m_tint.z << "]";
        return out.str();
    }
};

GTEST_TEST(OrenNayar, power_conservation) {
    optix::float3 white = { 1, 1, 1 };
    optix::float3 wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));
    for (float roughness : {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
        auto oren_nayar = OrenNayarWrapper(roughness);
        auto res = BSDFTestUtils::directional_hemispherical_reflectance_function(oren_nayar, wo, 2048u);
        EXPECT_FLOAT3_EQ_EPS(res.reflectance, white, 0.0002f) << oren_nayar.to_string();
    }
}

GTEST_TEST(OrenNayar, Helmholtz_reciprocity) {
    optix::float3 wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));
    for (float roughness : {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
        auto oren_nayar = OrenNayarWrapper(roughness);
        BSDFTestUtils::helmholtz_reciprocity(oren_nayar, wo, 16u);
    }
}

GTEST_TEST(OrenNayar, function_consistency) {
    optix::float3 wo = optix::normalize(optix::make_float3(1.0f, 1.0f, 1.0f));
    for (float roughness : {0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}) {
        auto oren_nayar = OrenNayarWrapper(roughness);
        BSDFTestUtils::BSDF_consistency_test(oren_nayar, wo, 16u);
    }
}

GTEST_TEST(OrenNayar, sampling_standard_deviation) {
    float roughness[5] = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };
    float expected_rho_std_devs[5] = { 0.0f, 0.074f, 0.095f, 0.114f, 0.135f };
    for (int i = 0; i < 5; i++) {
        auto oren_nayar = OrenNayarWrapper(roughness[i]);
        BSDFTestUtils::BSDF_sampling_variance_test(oren_nayar, 1024, expected_rho_std_devs[i]);
    }
}

GTEST_TEST(OrenNayar, input_albedo_equals_actual_reflectance) {
    optix::float3 albedo = { 0.25f, 0.5f, 0.75f };
    for (float roughness : {0.25f, 0.5f, 0.75f }) {
        auto oren_nayar = OrenNayarWrapper(roughness, albedo);
        for (float cos_theta_o : {0.1f, 0.5f, 0.9f }) {
            optix::float3 wo = BSDFTestUtils::w_from_cos_theta(cos_theta_o);
            auto reflectance = BSDFTestUtils::directional_hemispherical_reflectance_function(oren_nayar, wo, 2048).reflectance;
            EXPECT_FLOAT3_EQ_EPS(reflectance, albedo, 0.0002f) << oren_nayar.to_string();
        }
    }
}

GTEST_TEST(OrenNayar, E_approx_consistency) {
    for (float cos_theta : {0.1f, 0.5f, 0.9f }) {
        for (float roughness : {0.1f, 0.5f, 0.9f }) {
            float e_exact = Shading::BSDFs::OrenNayar::E_FON_exact(cos_theta, roughness);
            float e_approx = Shading::BSDFs::OrenNayar::E_FON_approx(cos_theta, roughness);
            EXPECT_FLOAT_EQ_EPS(e_exact, e_approx, 0.001f);
        }
    }
}

LineFitterf::Fit fit_line(OrenNayarWrapper brdf) {
    using namespace Bifrost::Math;

    LineFitterf fitter = LineFitterf();
    for (float cos_theta = 0.5f; cos_theta <= 0.95f; cos_theta += 0.01f) {
        optix::float3 wo = BSDFTestUtils::w_from_cos_theta(cos_theta);
        float f = brdf.evaluate(wo, wo).x * cos_theta;

        fitter.add_sample(cos_theta, f);
    }

    return fitter.fit();
}

void plot_brdf(Image image, OrenNayarWrapper brdf) {
    int width = image.get_width();
    int height = image.get_height();

    for (int x = 0; x < width; x++) {
        float cos_theta = (x + 0.5f) / width;
        optix::float3 wo = BSDFTestUtils::w_from_cos_theta(cos_theta);
        float f = brdf.evaluate(wo, wo).x * cos_theta;

        int y = int(f * (height - 1) + 0.5f);
        int y_min = max(0, y - 1);
        int y_max = min(height - 1, y + 1);
        for (int _y = y_min; _y <= y_max; _y++)
            image.set_pixel(RGBA(0, 1, 0, 1), Vector2ui(x, _y));
    }
}

void plot_line(Image image, Linef line) {
    int width = image.get_width();
    int height = image.get_height();
    for (int x = 0; x < width; x++) {
        float cos_theta = (x + 0.5f) / width;
        float f = line.evaluate(cos_theta);

        int y = int(f * (height - 1) + 0.5f);
        image.set_pixel(RGBA(1, 0, 1, 1), Vector2ui(x, y));
    }
}

float get_relative_intercept(Linef line) { return line.intercept / line.evaluate(1); }

GTEST_TEST(OrenNayar, plot_default_shading) {
    using namespace optix;
    using namespace Shading::ShadingModels;

    Images::allocate(8);

    Material material_params = {};
    material_params.tint = make_float3(1, 0, 0);
    material_params.specularity = 0.053f;

    for (float roughness : { 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f }) {
        material_params.roughness = roughness;

        int size = 512;

        // Create plot image
        Image image = Images::create2D("Plot", PixelFormat::RGB24, 2.2f, Vector2ui(size, size));
        image.clear();

        // Use red tint, then plot red as full BRDF and red - green as diffuse
        for (int x = 0; x < size; x++) {
            float cos_theta = (x + 0.5f) / size;
            optix::float3 wo = BSDFTestUtils::w_from_cos_theta(cos_theta);
            auto shading = DefaultShading(material_params, cos_theta);
            float3 reflectance = shading.evaluate_with_PDF(wo, wo).reflectance * cos_theta;

            float total_reflectance = reflectance.x;
            int total_y = int(total_reflectance * (size - 1) + 0.5f);
            total_y = min(size - 1, total_y);
            image.set_pixel(RGBA(1, 1, 1, 1), Vector2ui(x, total_y));

            float diffuse_reflectance = reflectance.x - reflectance.y;
            int diffuse_y = int(diffuse_reflectance * (size - 1) + 0.5f);
            diffuse_y = min(size - 1, diffuse_y);
            image.set_pixel(RGBA(0, 0, 1, 1), Vector2ui(x, diffuse_y));
        }

        // Save image with line info intercept + slope
        std::ostringstream name;
        name << "C:/Temp/DefaultShading/plot_roughness_" << roughness << ".png";
        StbImageWriter::write(image.get_ID(), name.str());
    }
}

GTEST_TEST(OrenNayar, roughness_to_lines) {
    Images::allocate(8);

    for (float roughness : { 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 2.0f }) {
        auto oren_nayar = OrenNayarWrapper(roughness);

        int size = 100;

        // Create plot image
        Image image = Images::create2D("Plot", PixelFormat::RGB24, 2.2f, Vector2ui(size, size));
        image.clear();

        plot_brdf(image, oren_nayar);

        auto fit = fit_line(oren_nayar);
        Linef line = fit.line;
        plot_line(image, line);

        // Save image with line info intercept + slope
        std::ostringstream name;
        name << "C:/Temp/OrenNayar/" << roughness << "_plot__" << line.slope << "x+" << line.intercept << "__incident_" << line.evaluate(1) << "__relIntercept_" << line.intercept / line.evaluate(1) << ".png";
        StbImageWriter::write(image.get_ID(), name.str());
    }
}

GTEST_TEST(OrenNayar, relative_intercept_to_roughness) {
    const float min_roughness = 0;
    const float max_roughness = 1.3f;

    for (float relative_intercept : { 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f }) {

        // Binary search for roughness matching relative intercept
        float low_roughness = min_roughness;
        float high_roughness = max_roughness;
        Linef low_fit_line = fit_line(OrenNayarWrapper(low_roughness)).line;
        Linef high_fit_line = fit_line(OrenNayarWrapper(high_roughness)).line;

        float middle_error = 1e6f;
        Linef middle_fit_line;
        for (int i = 0; i < 500; i++) {
            float middle_roughness = (low_roughness + high_roughness) / 2;
            middle_fit_line = fit_line(OrenNayarWrapper(middle_roughness)).line;
            middle_error = abs(relative_intercept - get_relative_intercept(middle_fit_line));
            float middle_relative_intercept = get_relative_intercept(middle_fit_line);

            if (middle_relative_intercept < relative_intercept) {
                low_roughness = middle_roughness;
                low_fit_line = middle_fit_line;
            } else {
                high_roughness = middle_roughness;
                high_fit_line = middle_fit_line;
            }
        }

        EXPECT_TRUE(false) << "Relative intercept: " << relative_intercept << " -> roughness: " << low_roughness << ", error: " << middle_error << "\n";
    }
}

GTEST_TEST(OrenNayar, validate_line_to_OrenNayar_parameters) {
    Images::allocate(8);

    auto relative_intercept_to_roughness = [](float relative_intercept) -> float {
        const int sample_count = 11;
        float intercept_to_roughness_map[sample_count] = {
            0.0f,
            0.084077119827270508f,
            0.17463022470474243f,
            0.27243340015411377f,
            0.37840759754180908f,
            0.4936220645904541f,
            0.61932891607284546f,
            0.75703203678131104f,
            0.90854072570800781f,
            1.0760570764541626f,
            1.262187123298645f,
        };

        float map_coord = relative_intercept * (sample_count - 1);
        int lower_map_index = int(map_coord);
        int upper_map_index = min(lower_map_index + 1, sample_count - 1);

        float lower_roughness = intercept_to_roughness_map[lower_map_index];
        float upper_roughness = intercept_to_roughness_map[upper_map_index];

        float t = map_coord - lower_map_index;
        return lerp(lower_roughness, upper_roughness, t);
    };

    auto relative_intercept_to_roughness_fit = [](float relative_intercept) -> float {
        return 0.549860f * relative_intercept * relative_intercept + 0.69475f * relative_intercept + 0.008021f;
    };

    for (float slope : { 0.01f, 0.25f, 0.5f, 0.75f, 1.0f }) {
        for (float intercept : { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f }) {

            Linef line = Linef(slope, intercept);
            float relative_intercept = line.intercept / line.evaluate(1); // TODO Handle nan

            // Map relative intercept to roughness using piece-wise linear function.
            float roughness = relative_intercept_to_roughness_fit(relative_intercept);

            // Scale BRDF by relative_intercept / actual_intercept
            float scale = intercept + slope;

            auto oren_nayar = OrenNayarWrapper(roughness, optix::make_float3(scale, scale, scale));

            // Plot
            int size = 100;
            Image image = Images::create2D("Plot", PixelFormat::RGB24, 2.2f, Vector2ui(size, size));
            RGB24* pixels = image.get_pixels<RGB24>();
            for (int i = 0; i < size * size; i++) {
                RGB24 c = { 0, 0, 0 };
                pixels[i] = c;
            }

            plot_brdf(image, oren_nayar);

            auto fit = fit_line(oren_nayar);
            plot_line(image, fit.line);

            // Save image with line info intercept + slope
            std::ostringstream name;
            name << "C:/Temp/OrenNayar/Validate_roughness" << roughness << "__line_" << line.slope << "x+" << line.intercept << "__incident_" << line.evaluate(1) << "__relIntercept_" << line.intercept / line.evaluate(1) << ".png";
            StbImageWriter::write(image.get_ID(), name.str());
        }
    }
}

} // NS OptiXRenderer

#endif // _OPTIXRENDERER_BSDFS_OREN_NAYAR_TEST_H_