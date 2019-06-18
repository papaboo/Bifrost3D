// Bifrost Matrix.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _BIFROST_MATH_MATRIX_H_
#define _BIFROST_MATH_MATRIX_H_

#include <Bifrost/Core/Defines.h>
#include <Bifrost/Math/Vector.h>

#include <cstring>
#include <initializer_list>
#include <sstream>
#include <type_traits>

namespace Bifrost {
namespace Math {

// ------------------------------------------------------------------------------------------------
// Row-major matrix representation.
// ------------------------------------------------------------------------------------------------
template <int R, int C, typename T>
struct Matrix final {
public:
    typedef typename T T;
    typedef typename T value_type;
    static const int ROW_COUNT = R;
    static const int COLUMN_COUNT = C;
    static const int ELEMENT_COUNT = ROW_COUNT * COLUMN_COUNT;
    static const int N = ROW_COUNT * COLUMN_COUNT;
    typedef typename Matrix<R, C, T> MatrixType;
    using RowType = typename std::conditional<COLUMN_COUNT == 4, Vector4<T>,
        typename std::conditional<COLUMN_COUNT == 3, Vector3<T>,
        typename std::conditional<COLUMN_COUNT == 2, Vector2<T>,
        std::false_type>::type>::type>::type;
    using HasRowType = std::enable_if_t<1 < C && C <= 4>;
    using ColumnType = typename std::conditional<ROW_COUNT == 4, Vector4<T>,
        typename std::conditional<ROW_COUNT == 3, Vector3<T>,
        typename std::conditional<ROW_COUNT == 2, Vector2<T>,
        std::false_type>::type>::type>::type;
    using HasColumnType = std::enable_if_t<1 < R && R <= 4>;

private:
    T m_elements[ELEMENT_COUNT];

public:

    //*********************************************************************************************
    // Constructors
    //*********************************************************************************************
    Matrix() = default;

    explicit Matrix(T v) {
        for (int i = 0; i < N; ++i)
            m_elements[i] = v;
    }

    Matrix(const std::initializer_list<T>& list) {
        if (N != list.size()) {
            printf("Initializer list size must match number of elements in matrix.\n");
            std::fill_n(m_elements, N, nanf(""));
        } else
            std::copy(list.begin(), list.end(), m_elements);
    }

    template<typename = HasRowType>
    Matrix(const std::initializer_list<RowType>& rows) {
        if (ROW_COUNT != rows.size()) {
            printf("Initializer list size must match number of rows in matrix.\n");
            std::fill_n(m_elements, N, nanf(""));
            return;
        }
        int r = 0;
        for (RowType row : rows) {
            std::copy(row.begin(), row.end(), m_elements + r * COLUMN_COUNT);
            ++r;
        }
    }

    template<typename U>
    Matrix(Matrix<R, C, U> rhs) {
        U* rhs_elements = rhs.begin();
        for (int i = 0; i < N; ++i)
            m_elements[i] = T(rhs_elements[i]);
    }

    //*****************************************************************************
    // Static constructor helpers.
    //*****************************************************************************
    static __always_inline__ MatrixType zero() {
        return MatrixType(0);
    }

    static __always_inline__ MatrixType identity() {
        MatrixType res;
        for (int r = 0; r < ROW_COUNT; ++r)
            for (int c = 0; c < COLUMN_COUNT; ++c)
                res[r][c] = r == c ? T(1) : T(0);
        return res;
    }

    //*****************************************************************************
    // Direct data access.
    // TODO Assert on indices and return checked RowView.
    //*****************************************************************************
    __always_inline__ T* begin() { return m_elements; }
    __always_inline__ const T* const begin() const { return m_elements; }
    __always_inline__ T* end() { return begin() + N; }
    __always_inline__ const T* const end() const { return begin() + N; }

    __always_inline__ T& operator()(int row, int column) { return m_elements[column + row * COLUMN_COUNT]; }
    __always_inline__ T operator()(int row, int column) const { return m_elements[column + row * COLUMN_COUNT]; }

    __always_inline__ T* operator[](int row) { return m_elements + row * COLUMN_COUNT; }
    __always_inline__ RowType const operator[](int row) const { return get_row(row); }

    //*****************************************************************************
    // Row and column getters and setters.
    //*****************************************************************************
    template<typename = HasRowType>
    __always_inline__ RowType get_row(int r) const {
        RowType row;
        memcpy(row.begin(), m_elements + r * COLUMN_COUNT, sizeof(RowType));
        return row;
    }
    template<typename = HasRowType>
    __always_inline__ void set_row(int r, RowType row) {
        memcpy(m_elements + r * COLUMN_COUNT, row.begin(), sizeof(RowType));
    }

    template<typename = HasColumnType>
    __always_inline__ ColumnType get_column(int c) const {
        ColumnType column;
        for (int r = 0; r < ROW_COUNT; ++r)
            column[r] = (*this)(r, c);
        return column;
    }
    template<typename = HasColumnType>
    __always_inline__ void set_column(int c, ColumnType column) {
        for (int r = 0; r < ROW_COUNT; ++r)
            (*this)(r, c) = column[r];
    }

    //*****************************************************************************
    // Multiplication operators
    //*****************************************************************************
    __always_inline__ MatrixType& operator*=(T rhs) {
        for (int i = 0; i < N; ++i)
            m_elements[i] *= rhs;
        return *this;
    }
    __always_inline__ MatrixType operator*(T rhs) const {
        MatrixType ret(*this);
        return ret *= rhs;
    }
    template <int RHS_ROW_COUNT>
    __always_inline__ Matrix<RHS_ROW_COUNT, COLUMN_COUNT, T> operator*(Matrix<RHS_ROW_COUNT, ROW_COUNT, T> rhs) const {
        Matrix<RHS_ROW_COUNT, COLUMN_COUNT, T> ret;
        for (int r = 0; r < ret.ROW_COUNT; ++r)
            for (int c = 0; c < ret.COLUMN_COUNT; ++c) {
                ret[r][c] = T(0);
                for (int d = 0; d < ROW_COUNT; ++d)
                    ret[r][c] += (*this)(r, d) * rhs[d][c];
            }
        return ret;
    }

    template<typename = HasRowType, typename = HasColumnType>
    __always_inline__ ColumnType operator*(RowType rhs) const {
        ColumnType res;
        for (int c = 0; c < ROW_COUNT; ++c)
            res[c] = dot(get_row(c), rhs);
        return res;
    }

    //*****************************************************************************
    // Division operators
    //*****************************************************************************
    __always_inline__ MatrixType& operator/=(T rhs) {
        for (int i = 0; i < N; ++i)
            m_elements[i] /= rhs;
        return *this;
    }
    __always_inline__ MatrixType operator/(T rhs) const {
        MatrixType ret(*this);
        return ret /= rhs;
    }

    //*****************************************************************************
    // Comparison operators.
    //*****************************************************************************
    __always_inline__ bool operator==(MatrixType rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) == 0;
    }
    __always_inline__ bool operator!=(MatrixType rhs) const {
        return memcmp(this, &rhs, sizeof(rhs)) != 0;
    }

    inline std::string to_string() const {
        std::ostringstream out;
        out << "[[";
        for (int e = 0; e < N; ++e) {
            out << m_elements[e];
            if (e % COLUMN_COUNT == COLUMN_COUNT)
                out << "], [";
            else if (e < N - 1)
                out << ",";
        }
        out << "]]";
        return out.str();
    }
};

//*************************************************************************
// Aliases and typedefs.
//*************************************************************************

template <typename T> using Matrix2x2 = Matrix<2, 2, T>;
template <typename T> using Matrix3x3 = Matrix<3, 3, T>;
template <typename T> using Matrix3x4 = Matrix<3, 4, T>;
template <typename T> using Matrix4x4 = Matrix<4, 4, T>;

using Matrix2x2f = Matrix2x2<float>;
using Matrix3x3f = Matrix3x3<float>;
using Matrix3x4f = Matrix3x4<float>;
using Matrix4x4f = Matrix4x4<float>;

// Compute the determinant of a 2x2 matrix.
template <typename T>
__always_inline__ T determinant(Matrix2x2<T> v) {
    return v[0][0] * v[1][1] - v[1][0] * v[0][1];
}

// Compute the determinant of a 3x3 matrix.
template <typename T>
__always_inline__ T determinant(Matrix3x3<T> v) {
    return v[0][0] * (v[1][1] * v[2][2] - v[1][2] * v[2][1])
        - v[0][1] * (v[1][0] * v[2][2] - v[1][2] * v[2][0])
        + v[0][2] * (v[1][0] * v[2][1] - v[1][1] * v[2][0]);
}

// Compute the determinant of a 4x4 matrix.
template <typename T>
inline T determinant(Matrix4x4<T> v) {
    return v[0][3] * v[1][2] * v[2][1] * v[3][0] - v[0][2] * v[1][3] * v[2][1] * v[3][0] - v[0][3] * v[1][1] * v[2][2] * v[3][0] + v[0][1] * v[1][3] * v[2][2] * v[3][0]
        + v[0][2] * v[1][1] * v[2][3] * v[3][0] - v[0][1] * v[1][2] * v[2][3] * v[3][0] - v[0][3] * v[1][2] * v[2][0] * v[3][1] + v[0][2] * v[1][3] * v[2][0] * v[3][1]
        + v[0][3] * v[1][0] * v[2][2] * v[3][1] - v[0][0] * v[1][3] * v[2][2] * v[3][1] - v[0][2] * v[1][0] * v[2][3] * v[3][1] + v[0][0] * v[1][2] * v[2][3] * v[3][1]
        + v[0][3] * v[1][1] * v[2][0] * v[3][2] - v[0][1] * v[1][3] * v[2][0] * v[3][2] - v[0][3] * v[1][0] * v[2][1] * v[3][2] + v[0][0] * v[1][3] * v[2][1] * v[3][2]
        + v[0][1] * v[1][0] * v[2][3] * v[3][2] - v[0][0] * v[1][1] * v[2][3] * v[3][2] - v[0][2] * v[1][1] * v[2][0] * v[3][3] + v[0][1] * v[1][2] * v[2][0] * v[3][3]
        + v[0][2] * v[1][0] * v[2][1] * v[3][3] - v[0][0] * v[1][2] * v[2][1] * v[3][3] - v[0][1] * v[1][0] * v[2][2] * v[3][3] + v[0][0] * v[1][1] * v[2][2] * v[3][3];
}

template <typename T>
__always_inline__ Matrix2x2<T> invert(Matrix2x2<T> v) {
    Matrix2x2<T> inverse;
    inverse[0][0] = v[1][1];
    inverse[0][1] = -v[0][1];
    inverse[1][0] = -v[1][0];
    inverse[1][1] = v[0][0];
    return inverse / determinant(v);
}

template <typename T>
inline Matrix3x3<T> invert(Matrix3x3<T> v) {
    Matrix3x3<T> inverse;

    inverse[0][0] = v[1][1] * v[2][2] - v[1][2] * v[2][1];
    inverse[0][1] = v[0][2] * v[2][1] - v[0][1] * v[2][2];
    inverse[0][2] = v[0][1] * v[1][2] - v[0][2] * v[1][1];

    inverse[1][0] = v[1][2] * v[2][0] - v[1][0] * v[2][2];
    inverse[1][1] = v[0][0] * v[2][2] - v[0][2] * v[2][0];
    inverse[1][2] = v[0][2] * v[1][0] - v[0][0] * v[1][2];

    inverse[2][0] = v[1][0] * v[2][1] - v[1][1] * v[2][0];
    inverse[2][1] = v[0][1] * v[2][0] - v[0][0] * v[2][1];
    inverse[2][2] = v[0][0] * v[1][1] - v[0][1] * v[1][0];

    return inverse / determinant(v);
}

template <typename T>
inline Matrix4x4<T> invert(Matrix4x4<T> v) {
    Matrix4x4<T> inverse;

    inverse[0][0] = v[1][2] * v[2][3] * v[3][1] - v[1][3] * v[2][2] * v[3][1] + v[1][3] * v[2][1] * v[3][2] - v[1][1] * v[2][3] * v[3][2] - v[1][2] * v[2][1] * v[3][3] + v[1][1] * v[2][2] * v[3][3];
    inverse[0][1] = v[0][3] * v[2][2] * v[3][1] - v[0][2] * v[2][3] * v[3][1] - v[0][3] * v[2][1] * v[3][2] + v[0][1] * v[2][3] * v[3][2] + v[0][2] * v[2][1] * v[3][3] - v[0][1] * v[2][2] * v[3][3];
    inverse[0][2] = v[0][2] * v[1][3] * v[3][1] - v[0][3] * v[1][2] * v[3][1] + v[0][3] * v[1][1] * v[3][2] - v[0][1] * v[1][3] * v[3][2] - v[0][2] * v[1][1] * v[3][3] + v[0][1] * v[1][2] * v[3][3];
    inverse[0][3] = v[0][3] * v[1][2] * v[2][1] - v[0][2] * v[1][3] * v[2][1] - v[0][3] * v[1][1] * v[2][2] + v[0][1] * v[1][3] * v[2][2] + v[0][2] * v[1][1] * v[2][3] - v[0][1] * v[1][2] * v[2][3];

    inverse[1][0] = v[1][3] * v[2][2] * v[3][0] - v[1][2] * v[2][3] * v[3][0] - v[1][3] * v[2][0] * v[3][2] + v[1][0] * v[2][3] * v[3][2] + v[1][2] * v[2][0] * v[3][3] - v[1][0] * v[2][2] * v[3][3];
    inverse[1][1] = v[0][2] * v[2][3] * v[3][0] - v[0][3] * v[2][2] * v[3][0] + v[0][3] * v[2][0] * v[3][2] - v[0][0] * v[2][3] * v[3][2] - v[0][2] * v[2][0] * v[3][3] + v[0][0] * v[2][2] * v[3][3];
    inverse[1][2] = v[0][3] * v[1][2] * v[3][0] - v[0][2] * v[1][3] * v[3][0] - v[0][3] * v[1][0] * v[3][2] + v[0][0] * v[1][3] * v[3][2] + v[0][2] * v[1][0] * v[3][3] - v[0][0] * v[1][2] * v[3][3];
    inverse[1][3] = v[0][2] * v[1][3] * v[2][0] - v[0][3] * v[1][2] * v[2][0] + v[0][3] * v[1][0] * v[2][2] - v[0][0] * v[1][3] * v[2][2] - v[0][2] * v[1][0] * v[2][3] + v[0][0] * v[1][2] * v[2][3];

    inverse[2][0] = v[1][1] * v[2][3] * v[3][0] - v[1][3] * v[2][1] * v[3][0] + v[1][3] * v[2][0] * v[3][1] - v[1][0] * v[2][3] * v[3][1] - v[1][1] * v[2][0] * v[3][3] + v[1][0] * v[2][1] * v[3][3];
    inverse[2][1] = v[0][3] * v[2][1] * v[3][0] - v[0][1] * v[2][3] * v[3][0] - v[0][3] * v[2][0] * v[3][1] + v[0][0] * v[2][3] * v[3][1] + v[0][1] * v[2][0] * v[3][3] - v[0][0] * v[2][1] * v[3][3];
    inverse[2][2] = v[0][1] * v[1][3] * v[3][0] - v[0][3] * v[1][1] * v[3][0] + v[0][3] * v[1][0] * v[3][1] - v[0][0] * v[1][3] * v[3][1] - v[0][1] * v[1][0] * v[3][3] + v[0][0] * v[1][1] * v[3][3];
    inverse[2][3] = v[0][3] * v[1][1] * v[2][0] - v[0][1] * v[1][3] * v[2][0] - v[0][3] * v[1][0] * v[2][1] + v[0][0] * v[1][3] * v[2][1] + v[0][1] * v[1][0] * v[2][3] - v[0][0] * v[1][1] * v[2][3];

    inverse[3][0] = v[1][2] * v[2][1] * v[3][0] - v[1][1] * v[2][2] * v[3][0] - v[1][2] * v[2][0] * v[3][1] + v[1][0] * v[2][2] * v[3][1] + v[1][1] * v[2][0] * v[3][2] - v[1][0] * v[2][1] * v[3][2];
    inverse[3][1] = v[0][1] * v[2][2] * v[3][0] - v[0][2] * v[2][1] * v[3][0] + v[0][2] * v[2][0] * v[3][1] - v[0][0] * v[2][2] * v[3][1] - v[0][1] * v[2][0] * v[3][2] + v[0][0] * v[2][1] * v[3][2];
    inverse[3][2] = v[0][2] * v[1][1] * v[3][0] - v[0][1] * v[1][2] * v[3][0] - v[0][2] * v[1][0] * v[3][1] + v[0][0] * v[1][2] * v[3][1] + v[0][1] * v[1][0] * v[3][2] - v[0][0] * v[1][1] * v[3][2];
    inverse[3][3] = v[0][1] * v[1][2] * v[2][0] - v[0][2] * v[1][1] * v[2][0] + v[0][2] * v[1][0] * v[2][1] - v[0][0] * v[1][2] * v[2][1] - v[0][1] * v[1][0] * v[2][2] + v[0][0] * v[1][1] * v[2][2];

    return inverse / determinant(v);
}

// Returns the matrix transposed.
template <int R, int C, typename T>
inline Matrix<C, R, T> transpose(Matrix<R, C, T> v) {
    Matrix<C, R, T> res;
    for (int r = 0; r < R; ++r)
        res.set_column(r, v.get_row(r));
    return res;
}

template <int R, int C, typename T, typename = Matrix<R, C, T>::RowType, typename = Matrix<R, C, T>::ColumnType>
inline typename Matrix<R, C, T>::RowType operator*(typename Matrix<R, C, T>::ColumnType lhs, Matrix<R, C, T> rhs) {
    Matrix<R, C, T>::RowType res;
    for (int c = 0; c < C; ++c)
        res[c] = dot(lhs, rhs.get_column(c));
    return res;
}

// Specialized multiplication operator for affine matrices. The bottom row is implicitly set to [0,0,0,1].
template <typename T>
__always_inline__ Matrix3x4<T> operator*(Matrix3x4<T> affine_lhs, Matrix3x4<T> affine_rhs) {
    Matrix3x4<T> res;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c) {
            res[r][c] = affine_lhs[r][0] * affine_rhs[0][c] + affine_lhs[r][1] * affine_rhs[1][c] + affine_lhs[r][2] * affine_rhs[2][c];
            if (c == 3)
                res[r][c] += affine_lhs[r][3];
        }
    return res;
}

template<int R, int C, typename T>
__always_inline__ bool almost_equal(Matrix<R, C, T> lhs, Matrix<R, C, T> rhs, unsigned short max_ulps = 4) {
    bool equal = true;
    for (int i = 0; i < R * C; ++i)
        equal &= almost_equal(lhs.begin()[i], rhs.begin()[i], max_ulps);
    return equal;
}

} // NS Math
} // NS Bifrost

  // Convenience function that appends a matrix' string representation to an ostream.
template<int R, int C, typename T>
__always_inline__ std::ostream& operator<<(std::ostream& s, Bifrost::Math::Matrix<R, C, T> v) {
    return s << v.to_string();
}

#endif // _BIFROST_MATH_MATRIX_H_
