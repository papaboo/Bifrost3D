// Precomputation of lookup tables for various BSDFs.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <FitRhoApproximation.h>
#include <GGXAlphaFromMaxPDF.h>
#include <PrecomputeDielectricBSDFRho.h>
#include <PrecomputeRoughBRDFRho.h>

#include <OptiXRenderer/Shading/BSDFs/Burley.h>
#include <OptiXRenderer/Shading/BSDFs/GGX.h>

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Math/RNG.h>

#include <StbImageWriter/StbImageWriter.h>

using namespace Bifrost::Assets;
using namespace optix;
using namespace OptiXRenderer;
using namespace OptiXRenderer::Shading::BSDFs;

int main(int argc, char** argv) {
    printf("Material Precomputations\n");

    std::string output_dir = argc >= 2? argv[1] : std::string(BIFROST_SHADING_DIR);
    printf("output_dir: %s\n", output_dir.c_str());

    const unsigned int width = 32, height = 32, sample_count = 4096;

    Images::allocate(1);
    Bifrost::Math::RNG::PmjbRNG rng(16384u);

    // fit_GGX_rho_approximation(output_dir);

    // Given a max PDF and cos(theta) compute the corresponding alpha of the GGX distribution with that max PDF.
    estimate_alpha_from_max_PDF(32, 32, output_dir + "EstimateGGXBoundedVNDFAlpha.cpp");

    { // Compute Burley rho.
        static auto sample_burley = [](float roughness, float3 wo, float2 random_sample) -> BSDFSample {
            float alpha = GGX::alpha_from_roughness(roughness);
            return Burley::sample({1, 1, 1}, alpha, wo, random_sample);
        };

        Image rho = PrecomputeRoughBRDFRho::tabulate_rho(width, height, sample_count, rng, sample_burley);

        // Store.
        StbImageWriter::write(rho, output_dir + "BurleyRho.png");
        PrecomputeRoughBRDFRho::output_brdf<1>(rho, sample_count, output_dir + "BurleyRho.cpp", "burley", "Directional-hemispherical reflectance for Burley.");
    }

    { // Compute GGX reflection rho.
        static auto sample_ggx = [](float roughness, float3 wo, float2 random_sample) -> BSDFSample {
            float alpha = GGX::alpha_from_roughness(roughness);
            return GGX_R::sample(alpha, 1, wo, random_sample);
        };

        Image rho = PrecomputeRoughBRDFRho::tabulate_rho(width, height, sample_count, rng, sample_ggx);

        // Store.
        StbImageWriter::write(rho, output_dir + "GGXRho.png");
        PrecomputeRoughBRDFRho::output_brdf<1>(rho, sample_count, output_dir + "GGXRho.cpp", "GGX", "Directional-hemispherical reflectance for GGX.");
    }

    { // Compute GGX reflection with fresnel rho.
        static auto sample_ggx_with_fresnel = [](float roughness, float3 wo, float2 random_sample) -> BSDFSample {
            float alpha = GGX::alpha_from_roughness(roughness);
            return GGX_R::sample(alpha, 0, wo, random_sample);
        };

        Image rho = PrecomputeRoughBRDFRho::tabulate_rho(width, height, sample_count, rng, sample_ggx_with_fresnel);

        // Store.
        StbImageWriter::write(rho, output_dir + "GGXWithFresnelRho.png");
        PrecomputeRoughBRDFRho::output_brdf<1>(rho, sample_count, output_dir + "GGXWithFresnelRho.cpp", "GGX_with_fresnel",
            "Directional-hemispherical reflectance for GGX with fresnel factor.");
    }

    { // Compute dielectric GGX rho.
        static auto sample_dielectric_ggx = [](float roughness, float ior_i, float3 wo, float3 random_sample) -> BSDFSample {
            float alpha = GGX::alpha_from_roughness(roughness);
            float specularity = OptiXRenderer::dielectric_specularity(1.0f, ior_i);
            return GGX::sample(alpha, specularity, ior_i, wo, random_sample);
        };

        int size = 16;
        auto rho = PrecomputeDielectricBSDFRho::tabulate_rho(size, size, size, sample_count, rng, sample_dielectric_ggx);

        // Store.
        PrecomputeDielectricBSDFRho::output_brdf(rho, output_dir + "DielectricGGXRho.cpp", "dielectric_GGX",
            "Directional-hemispherical reflectance for dielectric GGX.");
    }

    return 0;
}
