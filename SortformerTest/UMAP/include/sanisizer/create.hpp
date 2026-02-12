#ifndef SANISIZER_CREATE_HPP
#define SANISIZER_CREATE_HPP

#include <utility>

#include "utils.hpp"
#include "attest.hpp"

/**
 * @file create.hpp
 * @brief Safely create and resize containers.
 */

namespace sanisizer {

/**
 * Cast an integer to the type of the size of a container.
 * This protects against overflow when using this integer in the container's constructor or `resize()` method.
 *
 * @tparam Container_ Container class with a `size()` method and a constructor that accepts the size as the first argument.
 * @tparam Value_ Integer type of the input size.
 *
 * @param x Non-negative value representing the desired container size.
 *
 * @return `x` as the container's size's type.
 * If overflow would occur, an `OverflowError` is raised.
 */
template<class Container_, typename Value_>
constexpr auto as_size_type(Value_ x) {
    check_negative(x);
    typedef I<decltype(std::declval<Container_>().size())> Size;
    check_overflow<Size>(x);
    return static_cast<Size>(get_value(x));
}

/**
 * Create a new container of a specified size.
 * This protects against overflow when casting the integer size to the container's size type.
 *
 * @tparam Container_ Container class with a `size()` method and a constructor that accepts the size as the first argument.
 * @tparam Value_ Integer type of the input size.
 * @tparam Args_ Further arguments to pass to the container's constructor.
 *
 * @param x Non-negative value representing the desired container size.
 * @param args Additional arguments to pass to the `Container_` constructor after the size.
 *
 * @return An instance of the container, constructed to be of size `x`.
 * If overflow would occur, an `OverflowError` is raised.
 */
template<class Container_, typename Value_, typename ... Args_>
Container_ create(Value_ x, Args_&&... args) {
    return Container_(as_size_type<Container_>(x), std::forward<Args_>(args)...);
}

/**
 * Resize a container to the desired size.
 * This protects against overflow when casting the integer size to the container's size type.
 *
 * @tparam Container_ Container class with a `size()` method and a `resize()` method that accepts the size as the first argument.
 * @tparam Value_ Integer type of the input size.
 * @tparam Args_ Further arguments to pass to the container's `resize()` method.
 *
 * @param container An existing instance of the container.
 * This is resized to size `x`.
 * @param x Non-negative value representing the desired container size.
 * If this would overflow the container's size type, an `OverflowError` is raised.
 * @param args Additional arguments to pass to `resize()` after the size.
 */
template<class Container_, typename Value_, typename ... Args_>
void resize(Container_& container, Value_ x, Args_&&... args) {
    container.resize(as_size_type<Container_>(x), std::forward<Args_>(args)...);
}

}

#endif
