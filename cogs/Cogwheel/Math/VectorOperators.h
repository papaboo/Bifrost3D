//*****************************************************************************
// Header with operators for vectors.
// Used by Vector2, Vector3 and Vector4.
// Requires that 
// * the class is typedef'ed as Vector.
// * The number of elements in the Vector is given by N.
// * A pointer to the first element can be obtained by calling begin().
// * The type of each element is T.
//*****************************************************************************

//*****************************************************************************
// Indexing operators.
//*****************************************************************************
inline T& operator[](const int i) { return begin()[i]; }
inline T operator[](const int i) const { return begin()[i]; }

//*****************************************************************************
// Addition operators.
//*****************************************************************************
inline Vector<T>& operator+=(T rhs) {
    for (int i = 0; i < N; ++i)
        begin()[i] += rhs;
    return *this;
}
inline Vector<T>& operator+=(Vector<T> rhs) {
    for (int i = 0; i < N; ++i)
        begin()[i] += rhs[i];
    return *this;
}
inline Vector<T> operator+(T rhs) const {
    Vector<T> ret(*this);
    for (int i = 0; i < N; ++i)
        ret[i] += rhs;
    return ret;
}
inline Vector<T> operator+(Vector<T> rhs) const {
    Vector<T> ret(*this);
    for (int i = 0; i < N; ++i)
        ret[i] += rhs[i];
    return ret;
}

//*****************************************************************************
// Subtraction operators.
//*****************************************************************************
inline Vector<T>& operator-=(T rhs) {
    for (int i = 0; i < N; ++i)
        begin()[i] -= rhs;
    return *this;
}
inline Vector<T>& operator-=(Vector<T> rhs) {
    for (int i = 0; i < N; ++i)
        begin()[i] -= rhs[i];
    return *this;
}
inline Vector<T> operator-(T rhs) const {
    Vector<T> ret(*this);
    for (int i = 0; i < N; ++i)
        ret[i] -= rhs;
    return ret;
}
inline Vector<T> operator-(Vector<T> rhs) const {
    Vector<T> ret(*this);
    for (int i = 0; i < N; ++i)
        ret[i] -= rhs[i];
    return ret;
}
inline Vector<T> operator-() const {
    Vector<T> ret(*this);
    for (int i = 0; i < N; ++i)
        ret[i] = -(*this)[i];
    return ret;
}

//*****************************************************************************
// Multiplication operators.
//*****************************************************************************
inline Vector<T>& operator*=(T rhs) {
    for (int i = 0; i < N; ++i)
        begin()[i] *= rhs;
    return *this;
}
inline Vector<T>& operator*=(Vector<T> rhs) {
    for (int i = 0; i < N; ++i)
        begin()[i] *= rhs[i];
    return *this;
}
inline Vector<T> operator*(T rhs) const {
    Vector<T> ret(*this);
    for (int i = 0; i < N; ++i)
        ret[i] *= rhs;
    return ret;
}
inline Vector<T> operator*(Vector<T> rhs) const {
    Vector<T> ret(*this);
    for (int i = 0; i < N; ++i)
        ret[i] *= rhs[i];
    return ret;
}

//*****************************************************************************
// Division operators.
//*****************************************************************************
inline Vector<T>& operator/=(T rhs) {
    for (int i = 0; i < N; ++i)
        begin()[i] /= rhs;
    return *this;
}
inline Vector<T>& operator/=(Vector<T> rhs) {
    for (int i = 0; i < N; ++i)
        begin()[i] /= rhs[i];
    return *this;
}
inline Vector<T> operator/(T rhs) const {
    Vector<T> ret(*this);
    for (int i = 0; i < N; ++i)
        ret[i] /= rhs;
    return ret;
}
inline Vector<T> operator/(Vector<T> rhs) const {
    Vector<T> ret(*this);
    for (int i = 0; i < N; ++i)
        ret[i] /= rhs[i];
    return ret;
}

//*****************************************************************************
// Comparison operators.
//*****************************************************************************
inline bool operator==(Vector<T> rhs) const {
    return memcmp(this, &rhs, sizeof(rhs)) == 0;
}
inline bool operator!=(Vector<T> rhs) const {
    return memcmp(this, &rhs, sizeof(rhs)) != 0;
}