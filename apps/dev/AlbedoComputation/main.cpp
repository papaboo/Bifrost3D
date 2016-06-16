// Albedo or directional-hemispherical reflectance computation.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <OptiXRenderer/Shading/ShadingModels/DefaultShading.h>
#include <OptiXRenderer/RNG.h>

#include <Cogwheel/Assets/Image.h>
#include <Cogwheel/Core/Array.h>
#include <Cogwheel/Math/Utils.h>

#include <StbImageWriter/StbImageWriter.h>

#include <fstream>

using namespace Cogwheel;
using namespace Cogwheel::Assets;
using namespace optix;
using namespace OptiXRenderer;
using namespace OptiXRenderer::Shading::ShadingModels;

int main(int argc, char** argv) {

    // Compute the directional-hemispherical reflectance function, albedo, by monte carlo integration and store the result in a texture and as an array in a header file.
    // The diffuse and specular components are separated by tinting the diffuse layer with green and keeping the specular layer white.
    // The albedo is computed via monte arlo integration by assuming that the material is lit by a uniform infinitely far away area light with an intensity of one.
    // As the base material is green it has no contribution to the red and blue channels, which means that these contain the albedo of the specular component.
    // The green channel contains the contribution of both the specular and diffuse components and the diffuse contribution alone can be found by subtracting the specular contribution from the green channel.
    // Notes
    // * Fresnel base reflectivity is set to zero. This is completely unrealistic, but gives us the largest possible range between full diffuse and full specular.

    const unsigned int width = 64, height = 64, sample_count = 16384;

    Images::allocate(1);
    Image rho_image = Images::create("rho", PixelFormat::RGB_Float, 1.0f, Math::Vector2ui(width, height));
    Math::RGB* rho_image_pixels = (Math::RGB*)rho_image.get_pixels();
    
    // Specular material.
    Material material_params;
    material_params.base_tint = optix::make_float3(1.0f, 0.0f, 0.0f);
    material_params.metallic = 0.0f;
    material_params.specularity = 0.0f;

    for (int y = 0; y < int(height); ++y) {
        material_params.base_roughness = y / float(height - 1u);
        #pragma omp parallel for
        for (int x = 0; x < int(width); ++x) {

            float NdotV = fmaxf(x / float(width - 1u), 0.00001f);
            float3 wo = make_float3(sqrt(1.0f - NdotV * NdotV), 0.0f, NdotV);

            DefaultShading material = DefaultShading(material_params);

            // TODO Use hammersley RNG. See Unreal4 paper.
            RNG::LinearCongruential rng;
            rng.seed(294563u);

            Core::Array<double> specular_throughput = Core::Array<double>(sample_count);
            Core::Array<double> total_throughput = Core::Array<double>(sample_count);
            for (unsigned int s = 0; s < sample_count; ++s) {
                float3 rng_sample = rng.sample3f();
                
                BSDFSample sample = material.sample_all(wo, rng_sample);
                if (is_PDF_valid(sample.PDF)) {
                    total_throughput[s] = sample.weight.x * sample.direction.z / sample.PDF;
                    specular_throughput[s] = sample.weight.y * sample.direction.z / sample.PDF;
                } else
                    total_throughput[s] = specular_throughput[s] = 0.0;
            }

            double specular_rho = Math::sort_and_pairwise_summation(specular_throughput.begin(), specular_throughput.end()) / sample_count;
            double total_rho = Math::sort_and_pairwise_summation(total_throughput.begin(), total_throughput.end()) / sample_count;
            double diffuse_rho = total_rho - specular_rho;
            rho_image_pixels[x + y * width] = Math::RGB(float(diffuse_rho), float(specular_rho), 0.0f);
        }
    }

    // Store as image.
    StbImageWriter::write("DefaultShadingRho.png", rho_image);

    { // Store as header.
        std::ofstream out_header("DefaultShadingRho.h");
        out_header <<
            "// Directional-hemispherical reflectance for default shaded material.\n"
            "// ---------------------------------------------------------------------------\n"
            "// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors\n"
            "//\n"
            "// This program is open source and distributed under the New BSD License. See\n"
            "// LICENSE.txt for more detail.\n"
            "// ---------------------------------------------------------------------------\n"
            "// Generated by AlbedoComputation application.\n"
            "// ---------------------------------------------------------------------------\n"
            "\n"
            "#ifndef _OPTIXRENDERER_SHADING_SHADING_MODELS_DEFAULT_SHADING_RHO_H_\n"
            "#define _OPTIXRENDERER_SHADING_SHADING_MODELS_DEFAULT_SHADING_RHO_H_\n"
            "\n"
            "#include <optixu/optixu_math_namespace.h>\n"
            "\n"
            "namespace OptiXRenderer {\n"
            "\n"
            "const unsigned int default_shading_angle_sample_count = " << width << "u;\n"
            "const unsigned int default_shading_roughness_sample_count = " << height << "u;\n"
            "\n"
            "static const optix::float2 default_shading_rho[] = {\n";

        for (int y = 0; y < int(height); ++y) {
            float base_roughness = y / float(height - 1u);
            out_header << "    // Roughness " << base_roughness << "\n";
            out_header << "    ";
            for (int x = 0; x < int(width); ++x) {
                Math::RGB& rho = rho_image_pixels[x + y * width];
                out_header << "optix::make_float2(" << rho.r << "f, " << rho.g << "f), ";
            }
            out_header << "\n";
        }

        out_header <<
            "};\n"
            "\n"
            "} // NS OptiXRenderer\n"
            "\n"
            "#endif // _OPTIXRENDERER_SHADING_SHADING_MODELS_DEFAULT_SHADING_RHO_H_\n";

        out_header.close();
    }

    return 0;
}
