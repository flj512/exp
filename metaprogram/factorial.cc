#include<cstdio>

template<int N>
int factorial()
{
    if constexpr(N==0){
        return 1;
    }else{
        return N*factorial<N-1>();
    }
}

int main()
{
    const int N = 5;
    printf("%d! = %d\n",N,factorial<N>());
    return 0;
}