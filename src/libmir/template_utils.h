#ifndef _LIBMIR_TEMPLATE_UTILS_H_
#define _LIBMIR_TEMPLATE_UTILS_H_

// Used to prevent automatic template argument deduction
template <typename T>
struct dependent_type
{
    using type = T;
};

#endif
