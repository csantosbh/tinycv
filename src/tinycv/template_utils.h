#ifndef _TINYCV_TEMPLATE_UTILS_H_
#define _TINYCV_TEMPLATE_UTILS_H_

namespace tinycv
{

// Used to prevent automatic template argument deduction
template <typename T>
struct dependent_type
{
    using type = T;
};

}

#endif
