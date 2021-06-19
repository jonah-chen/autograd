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
    diff()
        : x(), dx() {}
    // basic constructor, dx=0
    diff(T x)
        : x(x), dx() {}
    // normal constructor
    diff(T x, T dx)
        : x(x), dx(dx) {}
    // copy constructor, comes with the copy = operator i think.
    diff(const diff& other)
        : x(other.x), dx(other.dx) {}
    // debug print operator
    friend std::ostream& operator<< (std::ostream& os, const diff& num)
    {
        os << "x: " << num.x << " , dx: " << num.dx;
        return os;
    }
    
    diff zero(void) { return diff(); }

    // unary negative
    diff operator-(void) { return diff(-x, -dx); }
    
    // comparison operators
    inline bool operator==(const diff& other) { return x == other.x; }
    inline bool operator<(const diff& other) { return x < other.x; }
    inline bool operator<=(const diff& other) { return (*this == other) or (*this < other); }
    inline bool operator>(const diff& other) { return !(*this <= other); }
    inline bool operator>=(const diff& other) { return !(*this < other); }
    inline bool operator!=(const diff& other) { return !(*this == other); }

    // 4 arithemetic operations
    diff operator+(const diff& other) { return diff(x + other.x, dx + other.dx); }
    diff operator-(const diff& other) { return diff(x - other.x, dx - other.dx); }
    diff operator*(const diff& other) { return diff(x * other.x, x * other.dx + other.x * dx); }
    diff operator/(const diff& other) { return diff(x / other.x, (other.x * dx - x * other.dx) / (other.x * other.x)); }
    
    // arithemetic with constants
    diff operator+(T other) { return diff(x + other, dx); }
    diff operator-(T other) { return diff(x - other, dx); }
    diff operator*(T other) { return diff(x * other, dx * other); }
    diff operator/(T other) { return diff(x / other, dx / other); }

    // arithemitic with constants on from the other side
    friend diff operator+(T lhs, const diff& rhs) { return diff(lhs + rhs.x, rhs.dx); }
    friend diff operator-(T lhs, const diff& rhs) { return diff(lhs - rhs.x, -rhs.dx); }
    friend diff operator*(T lhs, const diff& rhs) { return diff(lhs * rhs.x, lhs * rhs.dx); }
    friend diff operator/(T lhs, const diff& rhs) { return diff(lhs / rhs.x, -lhs * rhs.dx / (rhs.x * rhs.x)); }

    // common activation functions
    diff relu(void) { return x >= 0 ? *this : diff(); }
    
    diff sigmoid(void) 
    { 
        T tmp = (T)1 / (1 + ::exp(x));
        return diff(tmp, dx * tmp * (1 - tmp)); 
    }
    
    // exp
    diff log(void) { return diff(::log(x), dx / x); }
    diff exp(void) { return diff(::exp(x), dx * ::exp(x)); }
    diff pow(T power) { return power ? diff(::pow(x, power), dx * power * ::pow(x, power-1)) : diff((T)1); }

    // hyperbolic
    diff sinh(void) { return diff(::sinh(x), dx * ::cosh(x)); }
    diff cosh(void) { return diff(::cosh(x), dx * ::sinh(x)); }

    diff tanh(void) // also an actvation function 
    {
        T tmp = ::tanh(x);
        return diff(tmp, dx * (1 - (tmp * tmp))); 
    }

    
    // trig
    diff sin(void) { return diff(::sin(x), dx * ::cos(x)); }
    diff cos(void) { return diff(::cos(x), -dx * ::sin(x)); }
    
    diff tan(void) 
    {
        T tmp = ::tan(x);
        return diff(tmp, dx * (1 + (tmp * tmp)));
    }

};
