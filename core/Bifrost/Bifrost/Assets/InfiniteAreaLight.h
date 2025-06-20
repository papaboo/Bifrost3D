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
#include <Bifrost/Math/Distribution2D.h>

#include <memory>

namespace Bifrost {
namespace Assets {

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
// Future work:
// * Perhaps add it to Scene::LightSources.
// ------------------------------------------------------------------------------------------------
class InfiniteAreaLight {
private:
    Texture m_latlong;
    const Math::Distribution2D<float> m_distribution;

public:

    //*********************************************************************************************
    // Constructor.
    //*********************************************************************************************
    explicit InfiniteAreaLight(Texture latlong)
        : m_latlong(latlong)
        , m_distribution(Math::Distribution2D<double>(std::unique_ptr<float[]>(compute_PDF(m_latlong)).get(),
                                                      m_latlong.get_image().get_width(), m_latlong.get_image().get_height())) { }

    InfiniteAreaLight(TextureID latlong_ID, float* latlong_PDF)
        : m_latlong(latlong_ID)
        , m_distribution(Math::Distribution2D<double>(latlong_PDF, m_latlong.get_image().get_width(), m_latlong.get_image().get_height())) { }

    //*********************************************************************************************
    // Getters.
    //*********************************************************************************************
    inline Texture get_texture() const { return m_latlong; }
    inline Image get_image() const { return m_latlong.get_image(); }
    inline unsigned int get_width() const { return m_latlong.get_image().get_width(); }
    inline unsigned int get_height() const { return m_latlong.get_image().get_height(); }
    inline const float* const get_image_marginal_CDF() const { return m_distribution.get_marginal_CDF(); }
    inline const float* const get_image_conditional_CDF() const { return m_distribution.get_conditional_CDF(); }

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

    //*********************************************************************************************
    // Static utility functions.
    //*********************************************************************************************

    static inline float* compute_PDF(Texture latlong) {
        float* PDF = new float[latlong.get_image().get_pixel_count()];
        compute_PDF(latlong, PDF);
        return PDF;
    }

    // Computes the array of per pixel PDFs for an infinite area light. 
    // The importance is based on the average radiance of a pixel.
    // The PDF input must contain room for at least as many elements as there are pixels.
    static inline void compute_PDF(Texture latlong, float* PDF_result);
};

// ------------------------------------------------------------------------------------------------
// Infinite area light utilities.
// ------------------------------------------------------------------------------------------------

namespace InfiniteAreaLightUtils {

template <typename T>
struct IBLConvolution {
    T* Pixels;
    int Width;
    int Height;
    float Roughness;
    int sample_count;
};

template <typename T, typename F>
inline void convolute(const InfiniteAreaLight& light, IBLConvolution<T>* begin, IBLConvolution<T>* end, F color_conversion);

inline void convolute(const InfiniteAreaLight& light, IBLConvolution<Math::RGB>* begin, IBLConvolution<Math::RGB>* end) {
    convolute(light, begin, end, [](Math::RGB c) -> Math::RGB { return c; });
}

// Reconstructs the solid angle per pixel PDF from the CDFs.
// WARNING: The PDF has not been scaled by sin_theta. This can only be done when the final sample direction is known.
inline void reconstruct_solid_angle_PDF_sans_sin_theta(const InfiniteAreaLight& light, float* per_pixel_PDF);

} // NS InfiniteAreaLightUtils
} // NS Assets
} // NS Bifrost

#include <Bifrost/Assets/InfiniteAreaLight.inl>

#endif // _BIFROST_ASSETS_INFINITE_AREA_LIGHT_H_
