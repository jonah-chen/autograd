#pragma once
#include <cmath>
#include <iostream>

// differentiable class
template <typename T>
class diff
{
    T x, dx;
public:
    // default constructor
    __device__ __host__ diff()
        : x(), dx() {}
    // basic constructor, dx=0
    __device__ __host__ diff(T x)
        : x(x), dx() {}
    // normal constructor
    __device__ __host__ diff(T x, T dx)
        : x(x), dx(dx) {}

    // copy constructor, comes with the copy = operator i think.
    __device__ __host__ diff(const diff& other)
        : x(other.x), dx(other.dx) {}
    // debug print operator
    friend std::ostream& operator<< (std::ostream& os, const diff& num)
    {
        os << "x: " << num.x << " , dx: " << num.dx;
        return os;
    }
    
    // unary negative
    __device__ __host__ diff operator-(void) { return diff(-x, -dx); }
    
    // comparison operators
    __device__ __host__ inline bool operator==(const diff& other) { return x == other.x; }
    __device__ __host__ inline bool operator<(const diff& other) { return x < other.x; }
    __device__ __host__ inline bool operator<=(const diff& other) { return (*this == other) or (*this < other); }
    __device__ __host__ inline bool operator>(const diff& other) { return !(*this <= other); }
    __device__ __host__ inline bool operator>=(const diff& other) { return !(*this < other); }
    __device__ __host__ inline bool operator!=(const diff& other) { return !(*this == other); }

    // 4 arithemetic operations
    __device__ __host__ diff operator+(const diff& other) { return diff(x + other.x, dx + other.dx); }
    __device__ __host__ diff operator-(const diff& other) { return diff(x - other.x, dx - other.dx); }
    __device__ __host__ diff operator*(const diff& other) { return diff(x * other.x, x * other.dx + other.x * dx); }
    __device__ __host__ diff operator/(const diff& other) { return diff(x / other.x, (other.x * dx - x * other.dx) / (other.x * other.x)); }
    
    // arithemetic with constants
    __device__ __host__ diff operator+(T other) { return diff(x + other, dx); }
    __device__ __host__ diff operator-(T other) { return diff(x - other, dx); }
    __device__ __host__ diff operator*(T other) { return diff(x * other, dx * other); }
    __device__ __host__ diff operator/(T other) { return diff(x / other, dx / other); }

    __device__ __host__ // arithemitic with constants on from the other side
    __device__ __host__ friend diff operator+(T lhs, const diff& rhs) { return diff(lhs + rhs.x, rhs.dx); }
    __device__ __host__ friend diff operator-(T lhs, const diff& rhs) { return diff(lhs - rhs.x, -rhs.dx); }
    __device__ __host__ friend diff operator*(T lhs, const diff& rhs) { return diff(lhs * rhs.x, lhs * rhs.dx); }
    __device__ __host__ friend diff operator/(T lhs, const diff& rhs) { return diff(lhs / rhs.x, -lhs * rhs.dx / (rhs.x * rhs.x)); }

    // common activation functions
    __device__ __host__ diff relu(void) { return x >= 0 ? *this : diff(); }
    
    __device__ __host__ diff sigmoid(void) 
    { 
        T tmp = (T)1 / (1 + ::exp(x));
        return diff(tmp, dx * tmp * (1 - tmp)); 
    }
    
    // exp
    __device__ __host__ diff log(void) { return diff(::log(x), dx / x); }
    __device__ __host__ diff exp(void) { return diff(::exp(x), dx * ::exp(x)); }
    __device__ __host__ diff pow(T power) { return power ? diff(::pow(x, power), dx * power * ::pow(x, power-1)) : diff((T)1); }

    // hyperbolic
    __device__ __host__ diff sinh(void) { return diff(::sinh(x), dx * ::cosh(x)); }
    __device__ __host__ diff cosh(void) { return diff(::cosh(x), dx * ::sinh(x)); }

    __device__ __host__ diff tanh(void) // also an actvation function 
    {
        T tmp = ::tanh(x);
        return diff(tmp, dx * (1 - (tmp * tmp))); 
    }

    
    // trig
    __device__ __host__ diff sin(void) { return diff(::sin(x), dx * ::cos(x)); }
    __device__ __host__ diff cos(void) { return diff(::cos(x), -dx * ::sin(x)); }
    
    __device__ __host__ diff tan(void) 
    {
        T tmp = ::tan(x);
        return diff(tmp, dx * (1 + (tmp * tmp)));
    }

};
