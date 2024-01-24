// Fit the directional-hemispherical reflectance function (rho) to an apprimated model.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _FIT_RHO_APPROXIMATION_H_
#define _FIT_RHO_APPROXIMATION_H_

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Assets/Shading/Fittings.h>
#include <Bifrost/Math/NelderMead.h>
#include <Bifrost/Math/Vector.h>

#include <StbImageWriter/StbImageWriter.h>

using namespace Bifrost::Assets;
using namespace Bifrost::Assets::Shading;
using namespace Bifrost::Math;

// ------------------------------------------------------------------------------------------------
// Container for the two approximations fitted.
// ------------------------------------------------------------------------------------------------
struct RhoApproximation {
    float no_specularity;
    float full_specularity;
};

// ------------------------------------------------------------------------------------------------
// Model that approximates the directional-hemispherical reflectance function (rho) of GGX.
// Inspired by Treyarch and Unreal Engine 4.
// ------------------------------------------------------------------------------------------------
struct GgxRhoApproximation {

    // Rho approximation from Unreal Engine 4.
    // See https://www.unrealengine.com/en-US/blog/physically-based-shading-on-mobile
    static Vector3f evaluate_UE4(float roughness, float wo_dot_normal, Vector3f specularColor) {
        const Vector4f c0 = { -1.0f, -0.0275f, -0.572f, 0.022f };
        const Vector4f c1 = { 1.0f, 0.0425f, 1.04f, -0.04f };
        Vector4f r = roughness * c0 + c1;
        float a004 = fminf(r.x * r.x, exp2(-9.28f * wo_dot_normal)) * r.x + r.y;
        Vector2f AB = Vector2f(-1.04f, 1.04f) * a004 + Vector2f(r.z, r.w);
        return specularColor * AB.x + AB.y;
    }

    float m_params[11];

    GgxRhoApproximation(float* params) {
        std::copy(params, params + 11, m_params);
    }

    // Static constructor with Unreal Engine 4 parameters.
    static GgxRhoApproximation UE4() {
        float UE4_params[11] = { -1.0f, -0.0275f, -0.572f, 0.022f, 1.0f, 0.0425f, 1.04f, -0.04f, -9.28f, -1.04f, 1.04f };
        return GgxRhoApproximation(UE4_params);
    }

    // Evaluate the approximation.
    RhoApproximation evaluate(float roughness, float wo_dot_normal) {
        float* ps = m_params;

        const Vector4f c0 = { ps[0], ps[1], ps[2], ps[3] };
        const Vector4f c1 = { ps[4], ps[5], ps[6], ps[7] };
        Vector4f r = roughness * c0 + c1;
        float a004 = fminf(r.x * r.x, exp2(ps[8] * wo_dot_normal)) * r.x + r.y;
        Vector2f AB = Vector2f(ps[9], ps[10]) * a004 + Vector2f(r.z, r.w);

        float no_specularity = AB.y;
        float full_specularity = no_specularity + AB.x;
        return { no_specularity, full_specularity };
    }
};

// ------------------------------------------------------------------------------------------------
// Fits the Unreal Engine GGX Rho approximation to Bifrost's GGX implementation
// using the precomputed values for rho.
// ------------------------------------------------------------------------------------------------
struct GgxRhoFitter {
    enum class LossFunction {
        NoSpecularity = 1,
        FullSpecularity = 2,
        All = 3,
    };

    LossFunction m_loss_function = LossFunction::All;

    // Fits the Unreal Engine GGX Rho approximation to Bifrost's GGX implementation
    static std::pair<GgxRhoApproximation, float> fit(LossFunction loss_function) {
        float* start_params = GgxRhoApproximation::UE4().m_params;
        float end_params[11];

        GgxRhoFitter fitter;
        fitter.m_loss_function = loss_function;

        float loss = Bifrost::Math::nelder_mead<11>(end_params, start_params, 0.005f, 1e-5f, 5000, fitter);

        GgxRhoApproximation approximation(end_params);

        return { approximation, loss };
    }

    // Compute the loss of a GGX rho approximation
    static float loss(GgxRhoApproximation approx, LossFunction loss) {
        double accumulated_error = 0.0;
        for (int r = 0; r < Rho::GGX_roughness_sample_count; ++r)
        {
            float roughness = r / float(Rho::GGX_roughness_sample_count - 1);
            for (int a = 0; a < Rho::GGX_angle_sample_count; ++a)
            {
                float wo_dot_normal = (a + 0.5f) / float(Rho::GGX_angle_sample_count);

                RhoApproximation rho_approximation = approx.evaluate(roughness, wo_dot_normal);

                float no_specularity_gt = Rho::sample_GGX_with_fresnel(wo_dot_normal, roughness);
                float full_specularity_gt = Rho::sample_GGX(wo_dot_normal, roughness);

                if ((int(loss) & int(LossFunction::NoSpecularity)) != 0) {
                    float no_specularity_delta = no_specularity_gt - rho_approximation.no_specularity;
                    accumulated_error += no_specularity_delta * no_specularity_delta;
                }
                if ((int(loss) & int(LossFunction::FullSpecularity)) != 0) {
                    float full_specularity_delta = full_specularity_gt - rho_approximation.full_specularity;
                    accumulated_error += full_specularity_delta * full_specularity_delta;
                }
            }
        }

        return float(accumulated_error / (Rho::GGX_roughness_sample_count + Rho::GGX_angle_sample_count));
    }

    float operator()(float* params) {
        GgxRhoApproximation approx(params);
        return loss(approx, m_loss_function);
    }
};

void output_fitting_images(GgxRhoApproximation approximation, const std::string& output_dir, const std::string& prefix, int width = 64, int height = 64) {
    Image GGX_no_specularity_image = Images::create2D("GGX_no_specularity", PixelFormat::Intensity8, 2.2f, Vector2ui(width, height));
    Image GGX_full_specularity_image = Images::create2D("GGX_full_specularity", PixelFormat::Intensity8, 2.2f, Vector2ui(width, height));
    for (int y = 0; y < height; ++y)
    {
        float roughness = y / float(height - 1);
        for (int x = 0; x < width; ++x)
        {
            float wo_dot_normal = (x + 0.5f) / float(width);
            RhoApproximation rho = approximation.evaluate(roughness, wo_dot_normal);

            GGX_no_specularity_image.set_pixel(RGB(rho.no_specularity), Vector2ui(x, y));
            GGX_full_specularity_image.set_pixel(RGB(rho.full_specularity), Vector2ui(x, y));
        }
    }

    StbImageWriter::write(GGX_no_specularity_image, output_dir + prefix + "_GGX_no_specularity.png");
    StbImageWriter::write(GGX_full_specularity_image, output_dir + prefix + "_GGX_full_specularity.png");

    Images::destroy(GGX_no_specularity_image.get_ID());
    Images::destroy(GGX_full_specularity_image.get_ID());
}

void print_params(GgxRhoApproximation approximation) {
    float* ps = approximation.m_params;
    printf("    [%.4f, %.4f, %.4f, %.4f,\n", ps[0], ps[1], ps[2], ps[3]);
    printf("     %.4f, %.4f, %.4f, %.4f,\n", ps[4], ps[5], ps[6], ps[7]);
    printf("     %.4f, %.4f, %.4f]\n", ps[8], ps[9], ps[10]);
}

// Fits the Unreal Engine GGX Rho approximation to Bifrost's GGX implementation and outputs the result.
void fit_GGX_rho_approximation(const std::string& output_dir) {
    typedef GgxRhoFitter::LossFunction LossFunction;

    printf("Fit GGX directional-hemispherical reflectance function.\n");

    // Output original fitting as an image
    GgxRhoApproximation ue4_approximation = GgxRhoApproximation::UE4();
    output_fitting_images(ue4_approximation, output_dir, "UE4", 256, 256);
    float ue4_error = GgxRhoFitter::loss(ue4_approximation, LossFunction::All);
    printf("  UE4 loss %f\n", ue4_error);
    print_params(ue4_approximation);
    printf("\n");

    // Fit UE4 GGX rho approximation to our BRDF.
    for (LossFunction loss : { LossFunction::NoSpecularity, LossFunction::FullSpecularity, LossFunction::All})
    {
        std::pair<GgxRhoApproximation, float> rho_approximation_with_loss = GgxRhoFitter::fit(loss);

        std::string loss_name = "All";
        if (loss == LossFunction::NoSpecularity)
            loss_name = "NoSpecularity";
        else if (loss == LossFunction::FullSpecularity)
            loss_name = "FullSpecularity";

        // Output fit and the corresponding loss
        output_fitting_images(rho_approximation_with_loss.first, output_dir, "fitted_" + loss_name, 256, 256);
        printf("  fitted %s loss %f\n", loss_name.c_str(), rho_approximation_with_loss.second);
        print_params(rho_approximation_with_loss.first);
        printf("\n");
    }
}

#endif // _FIT_RHO_APPROXIMATION_H_