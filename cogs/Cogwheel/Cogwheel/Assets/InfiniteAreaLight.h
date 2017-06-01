// Cogwheel infinite area light.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_ASSETS_INFINITE_AREA_LIGHT_H_
#define _COGWHEEL_ASSETS_INFINITE_AREA_LIGHT_H_

#include <Cogwheel/Assets/Texture.h>
#include <Cogwheel/Math/Distribution2D.h>

#include <memory>

namespace Cogwheel {
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
    mutable TextureND m_latlong;
    const Math::Distribution2D<float> m_distribution;

public:

    //*********************************************************************************************
    // Constructor.
    //*********************************************************************************************
    explicit InfiniteAreaLight(Textures::UID latlong_ID)
        : m_latlong(latlong_ID)
        , m_distribution(Math::Distribution2D<double>(std::unique_ptr<float[]>(compute_PDF(m_latlong)).get(),
                                                      m_latlong.get_image().get_width(), m_latlong.get_image().get_height())) { }

    InfiniteAreaLight(Textures::UID latlong_ID, float* latlong_PDF)
        : m_latlong(latlong_ID)
        , m_distribution(Math::Distribution2D<double>(latlong_PDF, m_latlong.get_image().get_width(), m_latlong.get_image().get_height())) { }

    //*********************************************************************************************
    // Getters.
    //*********************************************************************************************
    inline Textures::UID get_texture_ID() const { return m_latlong.get_ID(); }
    inline Images::UID get_image_ID() const { return m_latlong.get_image().get_ID(); }
    inline unsigned int get_width() const { return m_latlong.get_image().get_width(); }
    inline unsigned int get_height() const { return m_latlong.get_image().get_height(); }
    inline const float* const get_image_marginal_CDF() const { return m_distribution.get_marginal_CDF(); }
    inline const float* const get_image_conditional_CDF() const { return m_distribution.get_conditional_CDF(); }

    //*********************************************************************************************
    // Evaluate.
    //*********************************************************************************************

    Math::RGB evaluate(Math::Vector2f uv) const {
        return sample2D(m_latlong.get_ID(), uv).rgb();
    }

    Math::RGB evaluate(Math::Vector3f direction_to_light) const {
        Math::Vector2f uv = Math::direction_to_latlong_texcoord(direction_to_light);
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
        sample.radiance = sample2D(m_latlong.get_ID(), CDF_sample.index).rgb();
        float sin_theta = abs(sqrtf(1.0f - sample.direction_to_light.y * sample.direction_to_light.y));
        float PDF = float(CDF_sample.PDF) / (2.0f * Math::PI<float>() * Math::PI<float>() * sin_theta);
        sample.PDF = sin_theta == 0.0f ? 0.0f : PDF;
        return sample;
    }

    float PDF(Math::Vector3f direction_to_light) const {
        float sin_theta = abs(sqrtf(1.0f - direction_to_light.y * direction_to_light.y));
        Math::Vector2f uv = Math::direction_to_latlong_texcoord(direction_to_light);
        float distribution_PDF = float(m_distribution.PDF_continuous(uv));
        float PDF = distribution_PDF / (2.0f * Math::PI<float>() * Math::PI<float>() * sin_theta);
        return sin_theta == 0.0f ? 0.0f : PDF;
    }

    //*********************************************************************************************
    // Static utility functions.
    //*********************************************************************************************

    static float* compute_PDF(TextureND latlong) {
        float* PDF = new float[latlong.get_image().get_pixel_count()];
        compute_PDF(latlong, PDF);
        return PDF;
    }

    // Computes the array of per pixel PDFs for an infinite area light. 
    // The importance is based on the average radiance of a pixel.
    // The PDF input must contain room for at least as many elements as there are pixels.
    static void compute_PDF(TextureND latlong, float* PDF_result);
};

// ------------------------------------------------------------------------------------------------
// Infinite area light utilities.
// ------------------------------------------------------------------------------------------------

namespace InfiniteAreaLightUtils {

struct IBLConvolution {
    Cogwheel::Math::RGB* Pixels;
    int Width;
    int Height;
    float Roughness;
    int sample_count;
};

void Convolute(const InfiniteAreaLight& light, IBLConvolution* begin, IBLConvolution* end);

} // NS InfiniteAreaLightUtils
} // NS Assets
} // NS Cogwheel

#endif // _COGWHEEL_ASSETS_INFINITE_AREA_LIGHT_H_