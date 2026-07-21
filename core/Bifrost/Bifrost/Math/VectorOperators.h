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
__always_inline__ GPU_ENABLED T& operator[](const int i) { return begin()[i]; }
__always_inline__ GPU_ENABLED T operator[](const int i) const { return begin()[i]; }

//*****************************************************************************
// Addition operators.
//*****************************************************************************
__always_inline__ GPU_ENABLED Vector<T>& operator+=(T rhs) {
    for (int i = 0; i < N; ++i)
        begin()[i] += rhs;
    return *this;
}
__always_inline__ GPU_ENABLED Vector<T>& operator+=(Vector<T> rhs) {
    for (int i = 0; i < N; ++i)
        begin()[i] += rhs[i];
    return *this;
}
__always_inline__ GPU_ENABLED Vector<T> operator+(T rhs) const {
    Vector<T> ret(*this);
    for (int i = 0; i < N; ++i)
        ret[i] += rhs;
    return ret;
}
__always_inline__ GPU_ENABLED Vector<T> operator+(Vector<T> rhs) const {
    Vector<T> ret(*this);
    for (int i = 0; i < N; ++i)
        ret[i] += rhs[i];
    return ret;
}

//*****************************************************************************
// Subtraction operators.
//*****************************************************************************
__always_inline__ GPU_ENABLED Vector<T>& operator-=(T rhs) {
    for (int i = 0; i < N; ++i)
        begin()[i] -= rhs;
    return *this;
}
__always_inline__ GPU_ENABLED Vector<T>& operator-=(Vector<T> rhs) {
    for (int i = 0; i < N; ++i)
        begin()[i] -= rhs[i];
    return *this;
}
__always_inline__ GPU_ENABLED Vector<T> operator-(T rhs) const {
    Vector<T> ret(*this);
    for (int i = 0; i < N; ++i)
        ret[i] -= rhs;
    return ret;
}
__always_inline__ GPU_ENABLED Vector<T> operator-(Vector<T> rhs) const {
    Vector<T> ret(*this);
    for (int i = 0; i < N; ++i)
        ret[i] -= rhs[i];
    return ret;
}
__always_inline__ GPU_ENABLED Vector<T> operator-() const {
    Vector<T> ret(*this);
    for (int i = 0; i < N; ++i)
        ret[i] = -(*this)[i];
    return ret;
}

//*****************************************************************************
// Multiplication operators.
//*****************************************************************************
__always_inline__ GPU_ENABLED Vector<T>& operator*=(T rhs) {
    for (int i = 0; i < N; ++i)
        begin()[i] *= rhs;
    return *this;
}
__always_inline__ GPU_ENABLED Vector<T>& operator*=(Vector<T> rhs) {
    for (int i = 0; i < N; ++i)
        begin()[i] *= rhs[i];
    return *this;
}
__always_inline__ GPU_ENABLED Vector<T> operator*(T rhs) const {
    Vector<T> ret(*this);
    for (int i = 0; i < N; ++i)
        ret[i] *= rhs;
    return ret;
}
__always_inline__ GPU_ENABLED Vector<T> operator*(Vector<T> rhs) const {
    Vector<T> ret(*this);
    for (int i = 0; i < N; ++i)
        ret[i] *= rhs[i];
    return ret;
}

//*****************************************************************************
// Division operators.
//*****************************************************************************
__always_inline__ GPU_ENABLED Vector<T>& operator/=(T rhs) {
    for (int i = 0; i < N; ++i)
        begin()[i] /= rhs;
    return *this;
}
__always_inline__ GPU_ENABLED Vector<T>& operator/=(Vector<T> rhs) {
    for (int i = 0; i < N; ++i)
        begin()[i] /= rhs[i];
    return *this;
}
__always_inline__ GPU_ENABLED Vector<T> operator/(T rhs) const {
    Vector<T> ret(*this);
    for (int i = 0; i < N; ++i)
        ret[i] /= rhs;
    return ret;
}
__always_inline__ GPU_ENABLED Vector<T> operator/(Vector<T> rhs) const {
    Vector<T> ret(*this);
    for (int i = 0; i < N; ++i)
        ret[i] /= rhs[i];
    return ret;
}

//*****************************************************************************
// Comparison operators.
//*****************************************************************************
__always_inline__ GPU_ENABLED bool operator==(Vector<T> rhs) const {
    bool equal = true;
    for (int i = 0; i < N; ++i)
        equal &= begin()[i] == rhs[i];
    return equal;
}
__always_inline__ GPU_ENABLED bool operator!=(Vector<T> rhs) const {
    bool not_equal = false;
    for (int i = 0; i < N; ++i)
        not_equal |= begin()[i] != rhs[i];
    return not_equal;
}