// Progressive multijittered bluenoise random number generator
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _PMJB_RNG_H_
#define _PMJB_RNG_H_

#include <Bifrost/Math/RNG.h>

#include <optixu/optixu_math_namespace.h>

struct PmjbRNG {
    int m_max_sample_capacity;
    Bifrost::Math::Vector2f* m_samples;

    PmjbRNG(int max_sample_capacity) {
        m_max_sample_capacity = max_sample_capacity;
        m_samples = new Bifrost::Math::Vector2f[max_sample_capacity];
        Bifrost::Math::RNG::fill_progressive_multijittered_bluenoise_samples(m_samples, m_samples + max_sample_capacity);
    }
    PmjbRNG(PmjbRNG& other) = delete;
    PmjbRNG(PmjbRNG&& other) = default;

    PmjbRNG& operator=(PmjbRNG& rhs) = delete;
    PmjbRNG& operator=(PmjbRNG&& rhs) = default;

    ~PmjbRNG() { delete[] m_samples; }

    optix::float2 sample_2f(int i) const { return optix::make_float2(m_samples[i].x, m_samples[i].y); }
    optix::float3 sample_3f(int i, int max_sample_count) const { return optix::make_float3(sample_2f(i), (i + 0.5f) / max_sample_count); }
};

#endif // _PMJB_RNG_H_