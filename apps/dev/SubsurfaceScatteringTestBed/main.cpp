// Subsurface scattering testbed
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <Bifrost/Assets/Image.h>

#include <OptiXRenderer/RNG.h>
#include <OptiXRenderer/Shading/BSDFs/BurleySSS.h>

#include <StbImageWriter/StbImageWriter.h>

using namespace Bifrost::Assets;
using namespace Bifrost::Math;
using namespace optix;
using namespace OptiXRenderer;
using namespace OptiXRenderer::Shading::BSDFs;

int main(int argc, char** argv) {
    printf("Subsurface scattering testbed\n");

    Images::allocate(1u);

    const int param_count = 5;
    auto ketchup_params = BurleySSS::Parameters::create({ 0.164f, 0.006f, 0.002f }, { 4.76f, 0.58f, 0.39f });
    auto marble_params = BurleySSS::Parameters::create({ 0.830f, 0.791f, 0.753f }, { 8.51f, 5.57f, 3.95f });
    auto potato = BurleySSS::Parameters::create({ 0.764f, 0.613f, 0.213f }, { 14.27f, 7.23f, 2.04f });
    auto skin1 = BurleySSS::Parameters::create({ 0.436f, 0.227f, 0.131f }, { 3.67f, 1.37f, 0.68f });
    auto whole_milk = BurleySSS::Parameters::create({ 0.908f, 0.881f, 0.759f }, { 10.90f, 6.58f, 2.51f });
    BurleySSS::Parameters bssrdf_params[param_count] = { potato, marble_params, whole_milk, skin1, ketchup_params };

    const int sample_count = 4092;
    
    int width = 100;
    int half_width = width / 2;
    int height = param_count * 20;
    Image image = Image::create2D("BurleySSS", PixelFormat::RGB24, true, Vector2ui(width, height));

    // Each pixel represents one mm in world space.
    // A 10 pixel wide column in the center of the image illuminates the plane consisting of the different materials.
    auto is_in_shadow = [=](float3 position) -> bool {
        return abs(position.x - half_width) > 5; // Pixels more than 5mm away from the center row are in shadow.
    };

    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 0; y < height; y++) {
        int bssrdf_index = int(y / (height / float(param_count)));
        BurleySSS::Parameters sss_params = bssrdf_params[bssrdf_index];
        for (int x = 0; x < width; x++) {
            int index_offset = RNG::teschner_hash(x, y);

            float3 radiance = { 0.0f, 0.0f, 0.0f };
            for (int i = 0; i < sample_count; i++) {
                auto rng = OptiXRenderer::RNG::PracticalScrambledSobol(index_offset + i, 0);
                float3 rng_sample = make_float3(rng.sample4f());

                auto sss_sample = BurleySSS::AlbedoMIS::sample(sss_params, make_float3(float(x), float(y), 0), rng_sample);
                bool in_shadow = is_in_shadow(sss_sample.position);
                if (!in_shadow)
                    radiance += sss_sample.reflectance / sss_sample.PDF.value();
            }

            radiance /= sample_count;
            RGBA pixel = { radiance.x, radiance.y, radiance.z, 1.0f };

            image.set_pixel(pixel, Vector2ui(x, y));
        }
    }

    StbImageWriter::write(image.get_ID(), "C:/Temp/BurleySSS.png");

    return 0;
}