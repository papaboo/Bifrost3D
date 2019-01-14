// Nelder-Mead-, inverse simplex-, amoebe optimization.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_MATH_NELDER_MEAD_H_
#define _BIFROST_MATH_NELDER_MEAD_H_

namespace Bifrost {
namespace Math {

// ------------------------------------------------------------------------------------------------
// Downhill simplex solver:
// http://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method#One_possible_variation_of_the_NM_algorithm
// using the termination criterion from Numerical Recipes in C++ (3rd Ed.)
// ------------------------------------------------------------------------------------------------
template<int DIM, typename Function>
float nelder_mead(float* pmin, const float* start, float delta, float tolerance, int max_iterations, Function objective_function) {

    // Standard coefficients from wiki page.
    const float reflect = 1.0f;
    const float expand = 2.0f;
    const float contract = 0.5f;
    const float shrink = 0.5f;

    typedef float Point[DIM];
    const int NB_POINTS = DIM + 1;

    auto copy = [](Point dst, const Point src) {
        for (int i = 0; i < DIM; ++i)
            dst[i] = src[i];
    };

    auto clear = [](Point r, float v = 0.0f) {
        for (int i = 0; i < DIM; ++i)
            r[i] = v;
    };

    auto add = [](Point r, const Point v) {
        for (int i = 0; i < DIM; ++i)
            r[i] += v[i];
    };

    Point s[NB_POINTS];
    float f[NB_POINTS];

    // Initialise simplex.
    copy(s[0], start);
    for (int i = 1; i < NB_POINTS; i++) {
        copy(s[i], start);
        s[i][i - 1] += delta;
    }

    // Evaluate function at each point on simplex.
    for (int i = 0; i < NB_POINTS; i++)
        f[i] = objective_function(s[i]);

    int lo = 0, hi, nh;
    for (int j = 0; j < max_iterations; j++) {
        // Find lowest, highest and next highest.
        lo = hi = nh = 0;
        for (int i = 1; i < NB_POINTS; i++) {
            if (f[i] < f[lo])
                lo = i;
            if (f[i] > f[hi]) {
                nh = hi;
                hi = i;
            }
            else if (f[i] > f[nh])
                nh = i;
        }

        // Stop if we've reached the required tolerance level.
        float a = fabsf(f[lo]);
        float b = fabsf(f[hi]);
        if (2.0f * fabsf(a - b) < (a + b) * tolerance)
            break;

        // Compute centroid excluding the worst point.
        Point o;
        clear(o);
        for (int i = 0; i < NB_POINTS; i++) {
            if (i == hi) continue;
            add(o, s[i]);
        }

        for (int i = 0; i < DIM; i++)
            o[i] /= DIM;

        // Reflection.
        Point r;
        for (int i = 0; i < DIM; i++)
            r[i] = o[i] + reflect * (o[i] - s[hi][i]);

        float fr = objective_function(r);
        if (fr < f[nh]) {
            if (fr < f[lo]) {
                // Expansion.
                Point e;
                for (int i = 0; i < DIM; i++)
                    e[i] = o[i] + expand*(o[i] - s[hi][i]);

                float fe = objective_function(e);
                if (fe < fr) {
                    copy(s[hi], e);
                    f[hi] = fe;
                    continue;
                }
            }

            copy(s[hi], r);
            f[hi] = fr;
            continue;
        }

        // Contraction.
        Point c;
        for (int i = 0; i < DIM; i++)
            c[i] = o[i] - contract*(o[i] - s[hi][i]);

        float fc = objective_function(c);
        if (fc < f[hi]) {
            copy(s[hi], c);
            f[hi] = fc;
            continue;
        }

        // Reduction.
        for (int k = 0; k < NB_POINTS; k++) {
            if (k == lo) continue;
            for (int i = 0; i < DIM; i++)
                s[k][i] = s[lo][i] + shrink*(s[k][i] - s[lo][i]);
            f[k] = objective_function(s[k]);
        }
    }

    // Return best point and its value.
    copy(pmin, s[lo]);
    return f[lo];
}

} // NS Math
} // NS Bifrost

#endif // _BIFROST_MATH_NELDER_MEAD_H_
