// Cogwheel 2D distribution.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_DISTRIBUTION2D_H_
#define _COGWHEEL_MATH_DISTRIBUTION2D_H_

#include <Cogwheel/Math/Vector.h>

namespace Cogwheel {
namespace Math {

// ------------------------------------------------------------------------------------------------
// A discretized 2D distribution.
// ------------------------------------------------------------------------------------------------
template <typename T>
struct Distribution2D final {

    const int m_width, m_height;
    T m_integral;
    T* m_marginal_CDF;
    T* m_conditional_CDF;

public:

    //*********************************************************************************************
    // Constructors and destructors.
    //*********************************************************************************************
    template <typename U>
    Distribution2D(U* function, int width, int height)
        : m_width(width), m_height(height),
        m_marginal_CDF(new T[m_height + 1]), m_conditional_CDF(new T[(m_width + 1) * m_height]) {
        m_integral = compute_CDFs(function, m_width, m_height, m_marginal_CDF, m_conditional_CDF);
    }

    ~Distribution2D() {
        delete[] m_marginal_CDF;
        delete[] m_conditional_CDF;
    }

    //*********************************************************************************************
    // Getters and setters.
    //*********************************************************************************************

    inline int get_width() const { return m_width; }
    inline int get_height() const { return m_height; }

    inline const T* const get_marginal_CDF() { return m_marginal_CDF; }
    inline int get_marginal_CDF_size() { return m_height + 1; }

    inline const T* const get_conditional_CDF() { return m_conditional_CDF; }
    inline Vector2i get_conditional_CDF_size() { return Vector2i(m_width + 1, m_height); }

    //*********************************************************************************************
    // Sampling.
    //*********************************************************************************************

    Vector2i samplei(Vector2f random_sample) const {
        static auto binary_search = [](float random_sample, T* CDF, int element_count) -> int {
            int lowerbound = 0;
            int upperbound = element_count;
            while (lowerbound + 1 != upperbound) {
                int middlebound = (lowerbound + upperbound) / 2;
                T cdf = CDF[middlebound];
                if (random_sample < cdf)
                    upperbound = middlebound;
                else
                    lowerbound = middlebound;
            }
            return lowerbound;
        };

        int y = binary_search(random_sample.y, m_marginal_CDF, m_height);
        int x = binary_search(random_sample.x, m_conditional_CDF + y * (m_width + 1), m_width);

        return Vector2i(x, y);
    }

    // TODO samplef and PDF

    //*********************************************************************************************
    // CDF construction.
    //*********************************************************************************************

    template <typename U>
    static T compute_CDFs(U* function, int width, int height, T* marginal_CDF, T* conditional_CDF) {
        
        // Compute conditional CDF.
        // #pragma omp parallel for schedule(dynamic, 16)
        for (int y = 0; y < height; ++y) {
            U* function_row = function + y * width;
            T* conditional_CDF_row = conditional_CDF + y * (width + 1);
            conditional_CDF_row[0] = T(0.0);
            for (int x = 0; x < width; ++x)
                conditional_CDF_row[x + 1] = conditional_CDF_row[x] + function_row[x];
        }

        // Compute marginal CDF.
        marginal_CDF[0] = T(0.0);
        for (int y = 0; y < height; ++y)
            marginal_CDF[y + 1] = marginal_CDF[y] + conditional_CDF[(y + 1) * (width + 1) - 1];

        // Integral of the function.
        T integral = marginal_CDF[height] / (width * height);

        // Normalize marginal CDF.
        for (int y = 1; y < height; ++y)
            marginal_CDF[y] /= marginal_CDF[height];
        marginal_CDF[height] = 1.0;

        // Normalize conditional CDF.
        // #pragma omp parallel for schedule(dynamic, 16)
        for (int y = 0; y < height; ++y) {
            T* conditional_CDF_row = conditional_CDF + y * (width + 1);
            if (conditional_CDF_row[width] > 0.0f)
                for (int x = 1; x < width; ++x)
                    conditional_CDF_row[x] /= conditional_CDF_row[width];
            // Last value should always be one. Even in rows with no contribution.
            // This ensures that the binary search is well-defined and will never select the last element.
            conditional_CDF_row[width] = 1.0f;
        }

        return integral;
    }

};

} // NS Math
} // NS Cogwheel

#endif // _COGWHEEL_MATH_DISTRIBUTION2D_H_