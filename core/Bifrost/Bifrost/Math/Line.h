// Bifrost lines.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_MATH_LINE_H_
#define _BIFROST_MATH_LINE_H_

#include <Bifrost/Math/Vector.h>

namespace Bifrost::Math {

// ------------------------------------------------------------------------------------------------
// Line
// ------------------------------------------------------------------------------------------------
template <typename T>
struct Line final {
    typedef T value_type;

    // ****************************************************************************
    // Public members
    // ****************************************************************************
    T slope;
    T intercept;

    Line() = default;
    Line(T slope, T intercept) : slope(slope), intercept(intercept) {}
    template <typename U>
    explicit Line(const Line<U>& l) : slope(T(l.slope)), intercept(T(l.intercept)) {}
    Line(Vector2<T> point0, Vector2<T> point1) {
        Vector2<T> delta = point1 - point0;
        slope = delta.y / delta.x;
        intercept = -slope * point0.x + point0.y;
    }

    // Evaluate the line at position x.
    __always_inline__ T evaluate(T x) const { return slope * x + intercept; }

    __always_inline__ T signed_distance(T x, T y) const { return y - evaluate(x); }
    __always_inline__ T signed_distance(Vector2<T> point) const { return point.y - evaluate(point.x); }

    __always_inline__ T distance(T x, T y) const { return abs(signed_distance(x, y)); }
    __always_inline__ T distance(Vector2<T> point) const { return abs(signed_distance(point)); }

    template <typename InputItr>
    __always_inline__ T mse(InputItr begin, InputItr end) const {
        T summed_squared_error = T(0);
        int count = 0;
        while (begin != end) {
            T distance = signed_distance(*begin++);
            summed_squared_error += distance * distance;
            count++;
        }
        return summed_squared_error / count;
    }
};

// ------------------------------------------------------------------------------------------------
// Utility that fits lines to weighted data samples. Lines are fitted using least squares.
// ------------------------------------------------------------------------------------------------
template <typename T>
struct LineFitter final {
public:
    struct Fit final {
        const Line<T> line;
        const T mse;
        Fit(Line<T> line, T mse) : line(line), mse(mse) {}
    };

private:
    Vector2<T> m_sum; // Sum of points
    Vector2<T> m_sum_sqr; // Sum of points squared
    T m_sum_xy; // Sum of x * y
    T m_sum_weight;

public:
    Vector2f sum() { return m_sum; }
    T summed_weight() { return m_sum_weight; }

    LineFitter()
        : m_sum(Vector2<T>::zero()), m_sum_sqr(Vector2<T>::zero()), m_sum_xy(T(0)), m_sum_weight(T(0)) {}

    __always_inline__ void add_sample(T x, T y, T weight = 1) {
        m_sum.x += weight * x;
        m_sum.y += weight * y;
        m_sum_sqr.x += weight * x * x;
        m_sum_sqr.y += weight * y * y;
        m_sum_xy += weight * x * y;
        m_sum_weight += weight;
    }
    __always_inline__ void add_sample(Vector2<T> point, T weight = 1) { return add_sample(point.x, point.y, weight); }

    // Fit a line to the samples added.
    __always_inline__ Fit fit() const {
        // Fit the slope and the intercept
        T a = (m_sum_weight * m_sum_xy - m_sum.x * m_sum.y) / (m_sum_weight * m_sum_sqr.x - m_sum.x * m_sum.x);
        T b = (m_sum.y - a * m_sum.x) / m_sum_weight;

        // Expression found by expanding 1 / W * sum(w_i(a * x_i + b - y_i)^2)
        T sum_error = m_sum_sqr.y + a * a * m_sum_sqr.x + m_sum_weight * b * b + T(2) * (a * (b * m_sum.x - m_sum_xy) - b * m_sum.y);

        return Fit(Line<T>(a, b), sum_error / m_sum_weight);
    }

    // Utility method that fits a line to a set of sample points. </summary>
    template <typename InputItr>
    static __always_inline__ Fit fit(InputItr begin, InputItr end) {
        LineFitter<T> fitter = LineFitter<T>();
        while (begin != end)
            fitter.add_sample(*begin++);
        return fitter.fit();
    }
};

// ------------------------------------------------------------------------------------------------
// Typedefs.
// ------------------------------------------------------------------------------------------------

typedef Line<double> Lined;
typedef Line<float> Linef;
typedef LineFitter<double> LineFitterd;
typedef LineFitter<float> LineFitterf;

} // NS Bifrost::Math

#endif // _BIFROST_MATH_LINE_H_
