// Precomputation of lookup tables for various BSDFs.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <FitRhoApproximation.h>
#include <GGXAlphaFromMaxPDF.h>
#include <PrecomputeRoughBRDFRho.h>

#include <OptiXRenderer/Shading/BSDFs/Burley.h>
#include <OptiXRenderer/Shading/BSDFs/GGX.h>
#include <OptiXRenderer/Shading/ShadingModels/DefaultShading.h>
#include <OptiXRenderer/RNG.h>

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Core/Array.h>
#include <Bifrost/Math/Utils.h>

#include <StbImageWriter/StbImageWriter.h>

using namespace Bifrost;
using namespace Bifrost::Assets;
using namespace optix;
using namespace OptiXRenderer;
using namespace OptiXRenderer::Shading::BSDFs;
using namespace OptiXRenderer::Shading::ShadingModels;

int main(int argc, char** argv) {
    printf("Material Precomputations\n");

    std::string output_dir = argc >= 2? argv[1] : std::string(BIFROST_SHADING_DIR);
    printf("output_dir: %s\n", output_dir.c_str());

    const unsigned int width = 32, height = 32, sample_count = 4096;

    Images::allocate(1);

    // fit_GGX_rho_approximation(output_dir);

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

        Image rho = Image::create2D("rho", PixelFormat::RGB_Float, 1.0f, Math::Vector2ui(width, height));
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
                    if (sample.PDF.is_valid()) {
                        total_throughput[s] = sample.reflectance.x * sample.direction.z / sample.PDF.value();
                        specular_throughput[s] = sample.reflectance.y * sample.direction.z / sample.PDF.value();
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
        PrecomputeRoughBRDFRho::output_brdf<2>(rho, sample_count, output_dir + "DefaultShadingRho.cpp", "default_shading",
            "Directional-hemispherical reflectance for default shaded material.");
    }

    { // Compute Burley rho.
        static auto sample_burley = [](float roughness, float3 wo, float2 random_sample) -> BSDFSample {
            float alpha = GGX::alpha_from_roughness(roughness);
            return Burley::sample({1, 1, 1}, alpha, wo, random_sample);
        };

        Image rho = PrecomputeRoughBRDFRho::estimate_rho(width, height, sample_count, sample_burley);

        // Store.
        StbImageWriter::write(rho, output_dir + "BurleyRho.png");
        PrecomputeRoughBRDFRho::output_brdf<1>(rho, sample_count, output_dir + "BurleyRho.cpp", "burley", "Directional-hemispherical reflectance for Burley.");
    }

    { // Compute GGX reflection rho.

        static auto sample_ggx = [](float roughness, float3 wo, float2 random_sample) -> BSDFSample {
            float alpha = GGX::alpha_from_roughness(roughness);
            return GGX_R::sample(alpha, 1, wo, random_sample);
        };

        Image rho = PrecomputeRoughBRDFRho::estimate_rho(width, height, sample_count, sample_ggx);

        // Store.
        StbImageWriter::write(rho, output_dir + "GGXRho.png");
        PrecomputeRoughBRDFRho::output_brdf<1>(rho, sample_count, output_dir + "GGXRho.cpp", "GGX", "Directional-hemispherical reflectance for GGX.");
    }

    { // Compute GGX reflection with fresnel rho.

        static auto sample_ggx_with_fresnel = [](float roughness, float3 wo, float2 random_sample) -> BSDFSample {
            float alpha = GGX::alpha_from_roughness(roughness);
            return GGX_R::sample(alpha, 0, wo, random_sample);
        };

        Image rho = PrecomputeRoughBRDFRho::estimate_rho(width, height, sample_count, sample_ggx_with_fresnel);

        // Store.
        StbImageWriter::write(rho, output_dir + "GGXWithFresnelRho.png");
        PrecomputeRoughBRDFRho::output_brdf<1>(rho, sample_count, output_dir + "GGXWithFresnelRho.cpp", "GGX_with_fresnel",
            "Directional-hemispherical reflectance for GGX with fresnel factor.");
    }

    return 0;
}
