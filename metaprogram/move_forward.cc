#include <iostream>
#include <vector>
/*
Left Value:
An lvalue (short for locator value) represents an object that has a location in
memory. In other words, an lvalue refers to an object that can appear on the
left-hand side of an assignment.

Right Value:
An rvalue (short for read value) represents a temporary object or value that
does not have a specific location in memory. An rvalue can only appear on the
right-hand side of an assignment (hence the name "rvalue").
*/

// L/R value reference
void lr_value_reference() {
  int a = 1;
  int &b = a;
  // int& c = 10; //Error: non-const lvalue reference can't bind to rvalue.
  const int &c = 2;  // const lvalue reference can bind to rvalue.

  int &&d = 3;
  // int&& e = a; //Error: rvalue reference can't bind to lvalue.
}

// Passing L/R value reference
void func1(int &a) {}
void func2(int &&a) {}
void lr_value_reference_pass(int &a, int &&b) {
  // the L/R value reference itself is treated as an lvalue
  // this means a,b is int type from the logic aspect.
  // notes: in the assemble code, both a and b stores the int address on the
  // stack.

  // assign a/b to v1, v2.
  int v1 = a;
  int v2 = b;

  // create another two l value refence to a/b.
  // this is ok because a/b is int type.
  int &l1 = a;
  int &l2 = b;

  // both the following expression is error
  // because r value reference can only bind to rvalue, but
  // a/b is is int type and is lvalue.
  // int&& r1 = a;
  // int&& r2 = b;

  // call func1 is OK.
  func1(a);
  func1(b);

  // can't call func2, the reason is same as above.
  // func2(a);
  // func2(b);

  // std::move
  /*
      // Here is a typical (simplified) implementation of std::move
      namespace std {
          template <typename T>
          constexpr T&& move(T&& arg) noexcept {
              return static_cast<T&&>(arg);
          }
      }

      // The core idea behind std::move is that it is a cast that converts an
     object to an rvalue reference type.
      // so you can think std::move equals 'static_cast<T&&>(arg)'
  */
  func2(std::move(a));
  func2(std::move(b));

  // 5 is a rvalue
  func2(5);
}

// Universal Reference
/*
A Universal Reference (also known as a Forwarding Reference) is a reference that
can bind to both lvalues and rvalues, depending on the type of the argument
passed to the function.

Universal references are typically written as T&& in function templates,
but the key distinction here is how they behave differently from regular rvalue
references. The behavior is determined at compile-time by the type of the
argument passed to the function.
*/

// Example:
template <typename T>
void func(T &&arg) {
  // Determine whether we have an lvalue or rvalue
  if constexpr (std::is_lvalue_reference<T &&>::value) {
    std::cout << "Lvalue\n";
  } else {
    std::cout << "Rvalue\n";
  }
}

/*
How Type Deduction Works in the above Example:

When func(a) is called and a is an lvalue.
-> The type T is deduced as int&.
-> T&& becomes int& &&, which collapses to int& (i.e., an lvalue reference).
-> The function prints "Lvalue reference" because T&& is treated as an lvalue
reference.

When func(20) is called and 20 is an rvalue.
->The type T is deduced as int (without a reference).
->T&& becomes int&&, which is an rvalue reference.
->The function prints "Rvalue reference" because T&& is treated as an rvalue
reference.

The Reference Collapsing Rule:
C++ has a reference collapsing rule that defines how references to references
behave in template deduction. Specifically:

T& & collapses to T&.
T& && collapses to T&.
T&& & collapses to T&.
T&& && remains T&&.

The last rule is particularly important because it allows universal references
to behave as either lvalue references or rvalue references depending on the type
of argument passed.
*/

// Perfect Forward
/*
The std::forward function primarily used in perfect forwarding.
It allows you to forward arguments to another function while maintaining their
value category (whether they are lvalues or rvalues).

    // Here’s a simplified version of the std::forward implementation found in
the C++ Standard Library.

    namespace std {
        template <typename T>
        T&& forward(typename remove_reference<T>::type& arg) noexcept {
            return static_cast<T&&>(arg);
        }

        template <typename T>
        T&& forward(typename remove_reference<T>::type&& arg) noexcept {
            return static_cast<T&&>(arg);
        }
    }
*/

// Forwarding function Example.
template <typename T>
void wrapper(T &&arg) {
  // Forward the argument to another function, preserving its value category
  func(std::forward<T>(arg));
}

// A function that takes an rvalue reference
void func(int &&x) { std::cout << "Rvalue func: " << x << "\n"; }

// A function that takes an lvalue reference
void func(int &x) { std::cout << "Lvalue func: " << x << "\n"; }

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

// A full example to show how to perfect forwad parameters.

class A {
 public:
  // std::move(desc) here is needed, because desc is now lvalue, see `Passing
  // L/R value reference` section. otherwise, init desc_ will call the copy
  // function of string.
  A(std::string &key, std::string value, std::string &&desc)
      : key_(key), value_(value), desc_(std::move(desc)) {}

  void print() {
    std::cout << "------ A ------" << std::endl;
    std::cout << "key:" << key_ << std::endl;
    std::cout << "value:" << value_ << std::endl;
    std::cout << "desc:" << desc_ << std::endl;
  }

 private:
  std::string key_;
  std::string value_;
  std::string desc_;
};

template <typename T>
class Cache {
 public:
  ~Cache() {
    for (auto &p : buffer_) {
      operator delete(p);  // only free memory without calling destructor
    }
  }
  template <typename... Args>
  // should use 'Args&&' to enable perfect forward.
  T *create(Args &&...args) {
    void *buff;
    if (buffer_.empty()) {
      std::cout << "\n\tno cache, create a new one." << std::endl;
      buff = operator new(
          sizeof(T));  // only alloc memory without calling constructor
    } else {
      std::cout << "\n\tget from cache." << std::endl;
      buff = *buffer_.rbegin();
      buffer_.pop_back();
    }
    return new (buff) T(std::forward<Args>(
        args)...);  // only call the T's constructor on `buff`
  }

  void destory(T *t) {
    t->~T();
    buffer_.push_back(t);
  }

 private:
  std::vector<void *> buffer_;
};
int main() {
  std::string key = "num";
  std::string value = "5";
  std::string desc = "this is test program";

  Cache<A> cache;

  auto a = cache.create(key, value, std::move(desc));
  a->print();
  cache.destory(a);

  a = cache.create(key, value, std::move(desc));
  a->print();
  cache.destory(a);

  return 0;
}
