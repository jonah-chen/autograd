#pragma once

#define LOG(x) (x).log()
#define EXP(x) (x).exp()

#define RELU(x) (x).relu()
#define SIGMOID(x) (x).sigmoid()

#define SINH(x) (x).sinh()
#define COSH(x) (x).cosh()
#define TANH(x) (x).tanh()

#define SIN(x) (x).sin()
#define COS(x) (x).cos()
#define TAN(x) (x).tan()

// for functions that are very suboptimal
#ifndef NO_SLOW
#define CSCH(x) (1 / ((x).sinh()))
#define SECH(x) (1 / ((x).cosh()))
#define COTH(x) (1 / ((x).tanh()))

#define CSC(x) (1 / ((x).sin()))
#define SEC(x) (1 / ((x).cos()))
#define COT(x) (1 / ((x).tan()))

#define SQRT(x) (x).pow(0.5)
#define CBRT(x) (x).pow(1.0/3.0)
#endif

