#include <cstdio>
#include <tuple>
#include <utility>

template <int N>
int factorial_recursive() {
  if constexpr (N == 0) {
    return 1;
  } else {
    return N * factorial_recursive<N - 1>();
  }
}

/*
    why does the `factorial_for_impl` need a `std::index_sequence<I...>` type
   parameter but don't use it here? this is becuase
   `std::make_index_sequence<N>()` generate an object of
   std::index_sequence<0,1,2,3...,N-1> so that the compiler can deduce the
   template parameters(which is `template <std::size_t... I>`) of
   `factorial_for_impl` is template<0,1,2,...N-1>.
*/
template <std::size_t... I>
int factorial_for_impl(std::index_sequence<I...>) {
  int ret = 1;
  // Using a fold expression to expand the indices
  ((ret *= (I + 1)), ...);
  return ret;
}

template <int N>
int factorial_for() {
  return factorial_for_impl(std::make_index_sequence<N>());
}

int main() {
  const int N = 5;
  printf("Recursive: %d! = %d\n", N, factorial_recursive<N>());
  printf("For: %d! = %d\n", N, factorial_for<N>());
  return 0;
}