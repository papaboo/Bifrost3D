// Cogwheel polynomial fitting utilities.
// ----------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_POLYNOMIAL_H_
#define _COGWHEEL_MATH_POLYNOMIAL_H_

#include <Cogwheel/Core/Array.h>
#include <Cogwheel/Math/Vector.h>

namespace Cogwheel {
namespace Math {

// ---------------------------------------------------------------------------
// Polynomial wrapper.
// The number of coefficients is degree + 1, 
// fx a polynomial of degree 2 has 3 coefficients, s2 * x^2 + s1 * x + s0.
// ---------------------------------------------------------------------------
struct Polynomial {
    Core::Array<float> coefficients;

    explicit Polynomial(int degree)
        : coefficients(degree + 1) { }

    int degree() const { return coefficients.size() - 1; }

    float evaluate(float x) {
        float res = coefficients[0];
        for (unsigned int i = 1; i < coefficients.size(); ++i) {
            res += coefficients[i] * x;
            x *= x;
        }
        return res;
    }

    inline std::string to_string() const {
        std::ostringstream out;
        out << coefficients[0];
        if (coefficients.size() > 0)
            out << " + " << coefficients[1] << "x";
        for (unsigned int i = 2; i < coefficients.size(); ++i)
            out << " + " << coefficients[i] << "x^" << i;
        return out.str();
    }
};

// ---------------------------------------------------------------------------
// Perform polynomial fiting using least squares by guassian elimination.
// Largely inspired by https://www.youtube.com/watch?v=YKmCiQLNcPA,
// and https://en.wikipedia.org/wiki/Gaussian_elimination.
// ---------------------------------------------------------------------------
inline Polynomial polynomial_fit_2D(Vector2f* x_y_begin, Vector2f* x_y_end, int polynomial_degree) {

    int data_pair_count = int(x_y_end - x_y_begin);
    int equation_count = polynomial_degree + 1;

    // Store the values of sigma(xi),sigma(xi^2),sigma(xi^3)....sigma(xi^2n)
    double* sigmas = new double[2 * polynomial_degree + 1];
    for (int i = 0; i < 2 * polynomial_degree + 1; ++i) {
        sigmas[i] = 0;
        for (int j = 0; j < data_pair_count; ++j)
            sigmas[i] += pow(x_y_begin[j].x, i);
    }
    
    // B holds the augmented matrix [B|A]. https://en.wikipedia.org/wiki/Augmented_matrix
    int B_width = equation_count + 1;
    int B_height = equation_count;
    double* B = new double[B_width * B_height];
    for (int i = 0; i < equation_count; ++i)
        for (int j = 0; j < equation_count; ++j)
            B[i * B_width + j] = sigmas[i + j];
    
    delete[] sigmas;

    // Store the values of sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi) in the final column of B.
    for (int i = 0; i < polynomial_degree + 1; ++i) {
        B[i * B_width + equation_count] = 0;
        for (int j = 0; j < data_pair_count; ++j)
            B[i * B_width + equation_count] += pow(x_y_begin[j].x, i) * x_y_begin[j].y;
    }
    
    // Perform gaussian elimination, i.e. turn the matrix into one in row echelon form.
    // TODO move the row with the largest absolute value value to the pivot positioning.
    // This improves numerical stability, but is otherwise not needed.
    for (int i = 0; i < equation_count - 1; ++i)
        for (int k = i + 1; k < equation_count; ++k) {
            double t = B[k * B_width + i] / B[i * B_width + i];
            for (int j = 0; j <= equation_count; ++j)
                B[k * B_width + j] = B[k * B_width + j] - t*B[i * B_width + j];
        }

    // Back-substitution.
    Polynomial polynomial = Polynomial(polynomial_degree);
    for (int i = equation_count - 1; i >= 0; --i) {
        double a = B[i * B_width + equation_count];
        for (int j = i+1; j < equation_count; ++j)
            a -= B[i * B_width + j] * polynomial.coefficients[j];
        polynomial.coefficients[i] = float(a / B[i * B_width + i]);
    }

    delete[] B;

    return polynomial;
}

} // NS Math
} // NS Cogwheel

// Convenience function that appends a polynomial's string representation to an ostream.
template<class T>
inline std::ostream& operator<<(std::ostream& s, Cogwheel::Math::Polynomial v) {
    return s << v.to_string();
}

#endif // _COGWHEEL_MATH_POLYNOMIAL_H_