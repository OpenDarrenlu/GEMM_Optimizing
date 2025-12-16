# GEMM_Optimizing
a collection of gemm optimization

## References
1. https://github.com/tpoisonooo/how-to-optimize-gemm
2. https://github.com/Yinghan-Li/YHs_Sample/tree/master/cuda/gemm
3. https://github.com/xlite-dev/LeetCUDA/tree/main/kernels



## 重要问题记录：
1. 
模板不是普通函数，编译器不会像普通函数那样先编译实现文件再链接匹配符号。
模板的代码必须在实例化时（也就是编译调用它的那个编译单元）可见，编译器才能生成对应类型（比如 __half）的机器码。
如果你只在 .cu 里写了 template <typename T> void random_matrix(...) { ... }，而调用点在另一个 .cu/.cpp 里，编译器看不到实现，就不会自动生成 random_matrix<__half>，所以会 undefined reference。