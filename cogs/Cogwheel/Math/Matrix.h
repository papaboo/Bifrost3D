// Cogwheel Matrix.
// ----------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ----------------------------------------------------------------------------

#ifndef _COGWHEEL_MATH_MATRIX_H_
#define _COGWHEEL_MATH_MATRIX_H_

#include <Math/Vector.h>

#include <initializer_list>
#include <sstream>

namespace Cogwheel {
namespace Math {

template <typename Row, typename Column>
struct Matrix final {
public:
    typedef typename Column::value_type T;
    typedef typename Column::value_type value_type;
    typedef Row Row;
    typedef Column Column;
    static const int RowCount = Column::N;
    static const int ColumnCount = Row::N;
    static const int N = Row::N * Column::N;

private:
    Row mRows[RowCount];

public:
    
    //*****************************************************************************
    // Constructors
    //*****************************************************************************
    Matrix() { }

    Matrix(T v) {
        T* data = &mRows[0][0];
        for (int i = 0; i < N; ++i)
            data[i] = v;
    }

    Matrix(const std::initializer_list<T>& list) {
        // assert(N == list.size(), "Initializer list size must match number of elements in matrix.");
        if (N != list.size())
            printf("Initializer list size must match number of elements in matrix.\n");
        T* data = &mRows[0][0];
        std::copy(list.begin(), list.end(), data);
    }

    Matrix(const std::initializer_list<Row>& list) {
        // assert(ColumnCount == list.size(), "Initializer list size must match number of columns in matrix.");
        Row* data = &mRows[0];
        std::copy(list.begin(), list.end(), data);
    }

    //*****************************************************************************
    // Static constructor helpers.
    //*****************************************************************************
    static Matrix<Row, Column> zero() {
        return Matrix<Row, Column>(0);
    }
    
    static Matrix<Row, Row> identity() { // Only square matrices have an identity, hence why row and column have the same type.
        Matrix<Row, Row> res = zero();
        for (int r = 0; r < RowCount; ++r)
            res.mRows[r][r] = T(1);
        return res;
    }

    //*****************************************************************************
    // Direct data access.
    //*****************************************************************************
    inline T* begin() { return &mRows[0][0]; }
    inline const T* const begin() const { return &mRows[0][0]; }
    inline Row& operator[](int r) { return mRows[r]; }
    inline Row operator[](int r) const { return mRows[r]; }

    //*****************************************************************************
    // Row and column getters and setters.
    //*****************************************************************************
    inline Row getRow(int i) const {
        return mRows[i];
    }
    inline void setRow(int i, Row row) {
        mRows[i] = row;
    }

    inline Column getColumn(int i) const {
        Column column;
        for (int r = 0; r < RowCount; ++r)
            column[r] = mRows[r][i];
        return column;
    }
    inline void setColumn(int i, Column column) {
        for (int r = 0; r < RowCount; ++r)
            mRows[r][i] = column[r];
    }

    //*****************************************************************************
    // Multiplication operators
    //*****************************************************************************
    inline Matrix<Row, Column>& operator*=(T rhs) {
        for (int i = 0; i < N; ++i)
            begin()[i] *= rhs;
        return *this;
    }
    inline Matrix<Row, Column> operator*(T rhs) const {
        Matrix<Row, Column> ret(*this);
        return ret *= rhs;
    }
    template <typename RhsRow>
    inline Matrix<RhsRow, Column> operator*(Matrix<RhsRow, Row> rhs) const {
        Matrix<RhsRow, Column> ret;
        for (int r = 0; r < ret.RowCount; ++r)
            for (int c = 0; c < ret.ColumnCount; ++c)
                ret[r][c] = dot(mRows[r], rhs.getColumn(c));
        return ret;
    }
    inline Column operator*(Row rhs) const {
        Column res;
        for (int c = 0; c < RowCount; ++c)
            res[c] = dot(mRows[c], rhs);
        return res;
    }

    //*****************************************************************************
    // Division operators
    //*****************************************************************************
    inline Matrix<Row, Column>& operator/=(T rhs) {
        for (int i = 0; i < N; ++i)
            begin()[i] /= rhs;
        return *this;
    }
    inline Matrix<Row, Column> operator/(T rhs) const {
        Matrix<Row, Column> ret(*this);
        return ret /= rhs;
    }


    //*****************************************************************************
    // Comparison operators.
    //*****************************************************************************
    inline bool operator==(Matrix<Row, Column> rhs) const {
        for (int i = 0; i < N; ++i) {
            if (begin()[i] != rhs[i]) return false;
        }
        return true;
    }
    inline bool operator!=(Matrix<Row, Column> rhs) const {
        for (int i = 0; i < N; ++i)
            if (begin()[i] == rhs[i]) return false;
        return true;
    }

    const std::string toString() const {
        std::ostringstream out;
        out << "[";
        for (int r = 0; r < RowCount; ++r) {
            out << "[" << mRows[r][0];
            for (int c = 1; c < ColumnCount; ++c) {
                out << ", " << mRows[r][c];
            }
            out << "]";
        }
        out << "]";
        return out.str();
    }
};

//*************************************************************************
// Aliases and typedefs.
//*************************************************************************

template <typename T> using Matrix2x2 = Matrix<Vector2<T>, Vector2<T>>;
template <typename T> using Matrix3x3 = Matrix<Vector3<T>, Vector3<T>>;
template <typename T> using Matrix4x3 = Matrix<Vector4<T>, Vector3<T>>;
template <typename T> using Matrix4x4 = Matrix<Vector4<T>, Vector4<T>>;

using Matrix2x2f = Matrix2x2<float>;
using Matrix3x3f = Matrix3x3<float>;
using Matrix4x3f = Matrix4x3<float>;
using Matrix4x4f = Matrix4x4<float>;

// Compute the determinant of a 2x2 matrix.
template <typename T>
inline T determinant(Matrix2x2<T> v) {
    return v[0][0] * v[1][1] - v[1][0] * v[0][1];
}

// Compute the determinant of a 3x3 matrix.
template <typename T>
inline T determinant(Matrix3x3<T> v) {
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
inline Matrix2x2<T> invert(Matrix2x2<T> v) {
    Matrix2x2<T> inverse;
    inverse[0][0] =  v[1][1];
    inverse[0][1] = -v[0][1];
    inverse[1][0] = -v[1][0];
    inverse[1][1] =  v[0][0];
    return inverse / determinant(v);
}

template <typename T>
inline Matrix3x3<T> invert(Matrix3x3<T> v) {
    Matrix3x3<T> inverse;

    inverse[0][0] = v[1][1]*v[2][2] - v[1][2]*v[2][1];
    inverse[0][1] = v[0][2]*v[2][1] - v[0][1]*v[2][2];
    inverse[0][2] = v[0][1]*v[1][2] - v[0][2]*v[1][1];

    inverse[1][0] = v[1][2]*v[2][0] - v[1][0]*v[2][2];
    inverse[1][1] = v[0][0]*v[2][2] - v[0][2]*v[2][0];
    inverse[1][2] = v[0][2]*v[1][0] - v[0][0]*v[1][2];

    inverse[2][0] = v[1][0]*v[2][1] - v[1][1]*v[2][0];
    inverse[2][1] = v[0][1]*v[2][0] - v[0][0]*v[2][1];
    inverse[2][2] = v[0][0]*v[1][1] - v[0][1]*v[1][0];

    return inverse / determinant(v);
}

template <typename T>
inline Matrix4x4<T> invert(Matrix4x4<T> v) {
    Matrix4x4<T> inverse;

    inverse[0][0] = v[1][2]*v[2][3]*v[3][1] - v[1][3]*v[2][2]*v[3][1] + v[1][3]*v[2][1]*v[3][2] - v[1][1]*v[2][3]*v[3][2] - v[1][2]*v[2][1]*v[3][3] + v[1][1]*v[2][2]*v[3][3];
    inverse[0][1] = v[0][3]*v[2][2]*v[3][1] - v[0][2]*v[2][3]*v[3][1] - v[0][3]*v[2][1]*v[3][2] + v[0][1]*v[2][3]*v[3][2] + v[0][2]*v[2][1]*v[3][3] - v[0][1]*v[2][2]*v[3][3];
    inverse[0][2] = v[0][2]*v[1][3]*v[3][1] - v[0][3]*v[1][2]*v[3][1] + v[0][3]*v[1][1]*v[3][2] - v[0][1]*v[1][3]*v[3][2] - v[0][2]*v[1][1]*v[3][3] + v[0][1]*v[1][2]*v[3][3];
    inverse[0][3] = v[0][3]*v[1][2]*v[2][1] - v[0][2]*v[1][3]*v[2][1] - v[0][3]*v[1][1]*v[2][2] + v[0][1]*v[1][3]*v[2][2] + v[0][2]*v[1][1]*v[2][3] - v[0][1]*v[1][2]*v[2][3];

    inverse[1][0] = v[1][3]*v[2][2]*v[3][0] - v[1][2]*v[2][3]*v[3][0] - v[1][3]*v[2][0]*v[3][2] + v[1][0]*v[2][3]*v[3][2] + v[1][2]*v[2][0]*v[3][3] - v[1][0]*v[2][2]*v[3][3];
    inverse[1][1] = v[0][2]*v[2][3]*v[3][0] - v[0][3]*v[2][2]*v[3][0] + v[0][3]*v[2][0]*v[3][2] - v[0][0]*v[2][3]*v[3][2] - v[0][2]*v[2][0]*v[3][3] + v[0][0]*v[2][2]*v[3][3];
    inverse[1][2] = v[0][3]*v[1][2]*v[3][0] - v[0][2]*v[1][3]*v[3][0] - v[0][3]*v[1][0]*v[3][2] + v[0][0]*v[1][3]*v[3][2] + v[0][2]*v[1][0]*v[3][3] - v[0][0]*v[1][2]*v[3][3];
    inverse[1][3] = v[0][2]*v[1][3]*v[2][0] - v[0][3]*v[1][2]*v[2][0] + v[0][3]*v[1][0]*v[2][2] - v[0][0]*v[1][3]*v[2][2] - v[0][2]*v[1][0]*v[2][3] + v[0][0]*v[1][2]*v[2][3];

    inverse[2][0] = v[1][1]*v[2][3]*v[3][0] - v[1][3]*v[2][1]*v[3][0] + v[1][3]*v[2][0]*v[3][1] - v[1][0]*v[2][3]*v[3][1] - v[1][1]*v[2][0]*v[3][3] + v[1][0]*v[2][1]*v[3][3];
    inverse[2][1] = v[0][3]*v[2][1]*v[3][0] - v[0][1]*v[2][3]*v[3][0] - v[0][3]*v[2][0]*v[3][1] + v[0][0]*v[2][3]*v[3][1] + v[0][1]*v[2][0]*v[3][3] - v[0][0]*v[2][1]*v[3][3];
    inverse[2][2] = v[0][1]*v[1][3]*v[3][0] - v[0][3]*v[1][1]*v[3][0] + v[0][3]*v[1][0]*v[3][1] - v[0][0]*v[1][3]*v[3][1] - v[0][1]*v[1][0]*v[3][3] + v[0][0]*v[1][1]*v[3][3];
    inverse[2][3] = v[0][3]*v[1][1]*v[2][0] - v[0][1]*v[1][3]*v[2][0] - v[0][3]*v[1][0]*v[2][1] + v[0][0]*v[1][3]*v[2][1] + v[0][1]*v[1][0]*v[2][3] - v[0][0]*v[1][1]*v[2][3];

    inverse[3][0] = v[1][2]*v[2][1]*v[3][0] - v[1][1]*v[2][2]*v[3][0] - v[1][2]*v[2][0]*v[3][1] + v[1][0]*v[2][2]*v[3][1] + v[1][1]*v[2][0]*v[3][2] - v[1][0]*v[2][1]*v[3][2];
    inverse[3][1] = v[0][1]*v[2][2]*v[3][0] - v[0][2]*v[2][1]*v[3][0] + v[0][2]*v[2][0]*v[3][1] - v[0][0]*v[2][2]*v[3][1] - v[0][1]*v[2][0]*v[3][2] + v[0][0]*v[2][1]*v[3][2];
    inverse[3][2] = v[0][2]*v[1][1]*v[3][0] - v[0][1]*v[1][2]*v[3][0] - v[0][2]*v[1][0]*v[3][1] + v[0][0]*v[1][2]*v[3][1] + v[0][1]*v[1][0]*v[3][2] - v[0][0]*v[1][1]*v[3][2];
    inverse[3][3] = v[0][1]*v[1][2]*v[2][0] - v[0][2]*v[1][1]*v[2][0] + v[0][2]*v[1][0]*v[2][1] - v[0][0]*v[1][2]*v[2][1] - v[0][1]*v[1][0]*v[2][2] + v[0][0]*v[1][1]*v[2][2];

    return inverse / determinant(v);
}

// Returns the matrix transposed.
template <typename Row, typename Column>
inline Matrix<Column, Row> transpose(Matrix<Row, Column> v) {
    Matrix<Column, Row> res;
    for (int r = 0; r < Column::N; ++r)
        res.setColumn(r, v.getRow(r));
    return res;
}

template <typename Row, typename Column>
inline bool almostEqual(Matrix<Row, Column> lhs, Matrix<Row, Column> rhs, unsigned short maxUlps = 4) {
    bool equal = true;
    for (int i = 0; i < Matrix<Row, Column>::N; ++i)
        equal &= almostEqual(lhs.begin()[i], rhs.begin()[i], maxUlps);
    return equal;
}

} // NS Math
} // NS Cogwheel

// Convinience function that appends a vector's string representation to an ostream.
template<typename Row, typename Column>
inline std::ostream& operator<<(std::ostream& s, Cogwheel::Math::Matrix<Row, Column> v){
    return s << v.toString();
}

#endif // _COGWHEEL_MATH_MATRIX_H_