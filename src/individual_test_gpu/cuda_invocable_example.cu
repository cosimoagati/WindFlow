#include <cstdint>
#include <iostream>
#include <functional>
#include <typeinfo>
#include <string>

using namespace std;

template <typename F, typename... Args>
struct is_invocable :
	std::is_constructible<std::function<void(Args ...)>,
			      std::reference_wrapper<typename std::remove_reference<F>::type>>
{};

auto f = [] __device__ (int &x) { x = x * x; };

template<typename func_t>
int
g(func_t f, int x)
{
	cout << is_invocable<func_t, int&>::value << endl;
	cout << is_invocable<func_t, int&, string>::value << endl;
	f(x);
}

template<typename tuple_t, typename result_t, typename func_t>
class Example
{
	func_t f;
public:
	Example (func_t f) : f {f}
	{
		cout << is_invocable<func_t, int&>::value << endl;
		cout << is_invocable<func_t, int&, string&>::value << endl;
	}
};

int main()
{
	cout << is_invocable<decltype(f), int&>::value << endl;
	cout << is_invocable<decltype(f), string>::value << endl;
	Example<int, int, decltype(f)> e {f};
	return 0;
}
