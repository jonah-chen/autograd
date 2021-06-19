#include "diff.hpp"
#include "abbrev.hpp"

int main()
{
    auto y = diff(1.0, 1.0);
    auto u = ((y.pow(4) + 1) / ((y*y) + 1)).pow(5);
    std::cout << u << std::endl;

    auto t = diff(4.0, 1.0);
    auto d = SIN(t + COS(SQRT(t)));
    std::cout << d << std::endl;
    return 0;
}
