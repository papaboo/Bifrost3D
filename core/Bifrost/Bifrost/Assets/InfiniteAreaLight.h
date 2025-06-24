// Bifrost infinite area light.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_ASSETS_INFINITE_AREA_LIGHT_H_
#define _BIFROST_ASSETS_INFINITE_AREA_LIGHT_H_

#include <Bifrost/Assets/Texture.h>
#include <Bifrost/Math/Distributions.h>
#include <Bifrost/Math/Distribution2D.h>
#include <Bifrost/Math/Quaternion.h>
#include <Bifrost/Math/RNG.h>

#include <memory>

namespace Bifrost::Assets {

// ------------------------------------------------------------------------------------------------
// A single sample from a light source.
// ------------------------------------------------------------------------------------------------
struct LightSample {
    Math::RGB radiance;
    float PDF;
    Math::Vector3f direction_to_light;
    float distance;
};

// ------------------------------------------------------------------------------------------------
// Samplable, textured infinite area light.
// ------------------------------------------------------------------------------------------------
class InfiniteAreaLight {
private:
    Texture m_latlong;
    Math::Distribution2D<float> m_distribution;

public:

    //*********************************************************************************************
    // The non-uniform surface area of the pixels lead to large variance in cases where the height discretization is to large.
    // We avoid this variance by ensuring that the PDF 2d distribution has a minimal height
    //*********************************************************************************************
    static const unsigned int MINIMUM_PDF_HEIGHT = 128;

    //*********************************************************************************************
    // Constructor.
    //*********************************************************************************************
    explicit InfiniteAreaLight(Texture latlong);

    //*********************************************************************************************
    // Getters.
    //*********************************************************************************************
    inline Texture get_texture() const { return m_latlong; }
    inline Image get_image() const { return m_latlong.get_image(); }
    inline unsigned int get_width() const { return m_latlong.get_image().get_width(); }
    inline unsigned int get_height() const { return m_latlong.get_image().get_height(); }

    inline const float* const get_image_marginal_CDF() const { return m_distribution.get_marginal_CDF(); }
    inline const float* const get_image_conditional_CDF() const { return m_distribution.get_conditional_CDF(); }
    inline unsigned int get_PDF_width() const { return m_distribution.get_width(); }
    inline unsigned int get_PDF_height() const { return m_distribution.get_height(); }

    //*********************************************************************************************
    // Evaluate.
    //*********************************************************************************************

    Math::RGB evaluate(Math::Vector2f uv) const {
        return sample2D(m_latlong, uv).rgb();
    }

    Math::RGB evaluate(Math::Vector3f direction_to_light) const {
        Math::Vector2f uv = Math::direction_to_latlong_texcoord(direction_to_light);
        uv.y = Math::min(uv.y, Math::nearly_one);
        return evaluate(uv);
    }

    //*********************************************************************************************
    // Sampling.
    //*********************************************************************************************

    float image_integral() const { return m_distribution.get_integral(); }

    LightSample sample(Math::Vector2f random_sample) const {
        auto CDF_sample = m_distribution.sample_continuous(random_sample);

        LightSample sample;
        sample.direction_to_light = Math::latlong_texcoord_to_direction(CDF_sample.index);
        sample.distance = 1e30f;
        sample.radiance = sample2D(m_latlong, CDF_sample.index).rgb();
        float sin_theta = abs(sqrtf(1.0f - sample.direction_to_light.y * sample.direction_to_light.y));
        float PDF = float(CDF_sample.PDF) / (2.0f * Math::PI<float>() * Math::PI<float>() * sin_theta);
        sample.PDF = sin_theta == 0.0f ? 0.0f : PDF;
        return sample;
    }

    float PDF(Math::Vector3f direction_to_light) const {
        float sin_theta = abs(sqrtf(1.0f - direction_to_light.y * direction_to_light.y));
        Math::Vector2f uv = Math::direction_to_latlong_texcoord(direction_to_light);
        uv.y = Math::min(uv.y, Math::nearly_one);
        float distribution_PDF = float(m_distribution.PDF_continuous(uv));
        float PDF = distribution_PDF / (2.0f * Math::PI<float>() * Math::PI<float>() * sin_theta);
        return sin_theta == 0.0f ? 0.0f : PDF;
    }
};

// ------------------------------------------------------------------------------------------------
// Infinite area light utilities.
// ------------------------------------------------------------------------------------------------

namespace InfiniteAreaLightUtils {

// Reconstructs the solid angle per pixel PDF from the CDFs.
// WARNING: The PDF has not been scaled by sin_theta. This can only be done when the final sample direction is known.
void reconstruct_solid_angle_PDF_sans_sin_theta(const InfiniteAreaLight& light, float* per_pixel_PDF);

template <typename T>
struct IBLConvolution {
    T* Pixels;
    int Width;
    int Height;
    float Roughness;
    int sample_count;
};

template <typename T, typename F>
inline void convolute(const InfiniteAreaLight& light, IBLConvolution<T>* begin, IBLConvolution<T>* end, F color_conversion) {

    using namespace Bifrost::Math;
    using namespace Bifrost::Math::Distributions;

    int max_requested_sample_count = 0;
    for (IBLConvolution<T>* itr = begin; itr != end; ++itr)
        max_requested_sample_count = max(max_requested_sample_count, itr->sample_count);

    unsigned int light_sample_count = 4 * max_requested_sample_count;
    unsigned int max_ggx_sample_count = 4 * max_requested_sample_count;

    // Precompute blue noise random samples.
    unsigned int max_sample_count = max(light_sample_count, max_ggx_sample_count);
    Vector2f* rng_samples = new Vector2f[max_sample_count];
    RNG::fill_progressive_multijittered_bluenoise_samples(rng_samples, rng_samples + max_sample_count);

    // Precompute light samples.
    LightSample* light_samples = new LightSample[light_sample_count];
    #pragma omp parallel for schedule(dynamic, 16)
    for (int s = 0; s < int(light_sample_count); ++s)
        light_samples[s] = light.sample(rng_samples[s]);

    // Preallocate GGX samples.
    GGX::Sample* ggx_samples = new GGX::Sample[max_ggx_sample_count];

    for (; begin != end; ++begin) {

        int width = begin->Width, height = begin->Height;
        float roughness = begin->Roughness;
        float alpha = roughness * roughness;

        // Handle nearly specular case.
        if (alpha < 0.00000000001f) {
            Texture env_map = light.get_texture();
            #pragma omp parallel for schedule(dynamic, 16)
            for (int i = 0; i < width * height; ++i) {
                int x = i % width, y = i / width;
                begin->Pixels[x + y * width] = color_conversion(sample2D(env_map, Vector2f((x + 0.5f) / width, (y + 0.5f) / height)).rgb());
            }
            continue;
        }

        unsigned int ggx_sample_count = 4 * begin->sample_count;
        #pragma omp parallel for schedule(dynamic, 16)
        for (int s = 0; s < int(ggx_sample_count); ++s)
            ggx_samples[s] = GGX::sample(alpha, rng_samples[s]);

        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < width * height; ++i) {

            int x = i % width;
            int y = i / width;

            Vector2f up_uv = Vector2f((x + 0.5f) / width, (y + 0.5f) / height);
            Vector3f up_vector = latlong_texcoord_to_direction(up_uv);
            Quaternionf up_rotation = Quaternionf::look_in(up_vector);

            unsigned int light_sample_count = begin->sample_count / 2;
            unsigned int light_sample_offset = RNG::teschner_hash(x, y);
            RGB light_radiance = RGB::black();
            for (unsigned int s = 0; s < light_sample_count; ++s) {
                const LightSample& sample = light_samples[(s + light_sample_offset) % light_sample_count];
                if (sample.PDF < 0.000000001f)
                    continue;

                float cos_theta = fmaxf(dot(sample.direction_to_light, up_vector), 0.0f);
                float ggx_f = GGX::D(alpha, cos_theta);
                float ggx_PDF = ggx_f * cos_theta; // Inlined GGX::PDF(alpha, cos_theta);
                if (isnan(ggx_f))
                    continue;

                float mis_weight = RNG::power_heuristic(sample.PDF, ggx_PDF);
                light_radiance += sample.radiance * (mis_weight * ggx_f * cos_theta / sample.PDF);
            }
            light_radiance /= float(light_sample_count);

            unsigned int ggx_sample_count = begin->sample_count - light_sample_count;
            unsigned int ggx_sample_offset = RNG::teschner_hash(x, y, 1);
            RGB ggx_radiance = RGB::black();
            for (unsigned int s = 0; s < ggx_sample_count; ++s) {
                GGX::Sample sample = ggx_samples[(s + ggx_sample_offset) % ggx_sample_count];
                if (sample.PDF < 0.000000001f)
                    continue;

                sample.direction = normalize(up_rotation * sample.direction);
                float mis_weight = RNG::power_heuristic(sample.PDF, light.PDF(sample.direction));
                ggx_radiance += light.evaluate(sample.direction) * mis_weight;
            }
            ggx_radiance /= float(ggx_sample_count);

            RGB radiance = ggx_radiance + light_radiance;
            begin->Pixels[x + y * width] = color_conversion(radiance);
        }
    }

    delete[] ggx_samples;
    delete[] light_samples;
    delete[] rng_samples;
}

inline void convolute(const InfiniteAreaLight& light, IBLConvolution<Math::RGB>* begin, IBLConvolution<Math::RGB>* end) {
    convolute(light, begin, end, [](Math::RGB c) -> Math::RGB { return c; });
}

} // NS InfiniteAreaLightUtils
} // NS Bifrost::Assets

#endif // _BIFROST_ASSETS_INFINITE_AREA_LIGHT_H_
