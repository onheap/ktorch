# Steps


## Set up development environment

### User Java incubator module (in Kotlin gradle project)
1. https://foojay.io/today/how-to-run-the-java-incubator-module-from-the-command-line-and-intellij-idea/
2. https://stackoverflow.com/questions/70390734/how-can-i-use-the-incubating-vector-api-from-kotlin-in-a-gradle-build



## Implement Numpy 


### Numpy Basic Operations
Implement Numpy _Broadcasting_ and _Dot Product_ in pure Java/Kotlin, and use SIMD to accelerate computation.

```python
# test case for broadcasting

# test case for dot product

# test case for broadcasting & dot product
```

### Prompt

我要写用 Java 写一个类似 Numpy 的项目, 需要能够实现类似 Numpy 中的 broadcast 以及 dot product 的功能, 具体的要求如下
1. 项目中的数据结构需要能够支持多维数组, 但只用支持 float32 类型即可
2. 项目中的数据结构需要能够支持 broadcast, 即支持不同维度的数组之间的运算
3. 项目中的数据结构需要能够支持 dot product, 即支持不同维度的数组之间的运算
4. 项目中的数组需要用 strides 格式保存, 以节省内存
5. 不要使用 Java Native API, 使用纯 Java 代码实现, 但可以使用 Java 的 incubator 中的 Vector API 来使用 SIMD 加速计算

### References
* https://ajcr.net/stride-guide-part-1/
* https://numpy.org/doc/1.20/reference/internals.html
* https://numpy.org/doc/1.20/reference/arrays.ndarray.html

#### Broadcasting
* https://numpy.org/doc/stable/user/basics.broadcasting.html

#### Dot Product
* https://www.reddit.com/r/java/comments/17pkcgx/how_fast_can_we_do_matrix_multiplications_in_pure/
* https://www.elastic.co/blog/accelerating-vector-search-simd-instructions
* Whether we need to use _Virtual Thread_ to parallelize Matrix Multiplication? Answer is NO. Virtual Thread is more suitable for I/O bound tasks. And Matrix Multiplication is a CPU bound task.  [ref](https://www.reddit.com/r/java/comments/16mkm4v/efficiency_of_java_21_virtual_threads_compared_to/)
* https://github.com/lessthanoptimal/VectorPerformance/blob/master/src/main/java/benchmark/MatrixMultiplication.java
* https://github.com/openjdk/jdk/pull/15338
* https://www.baeldung.com/jvm-tiered-compilation


## Implement Simple NN

https://github.com/tinygrad/tinygrad/blob/91a352a8e2697828a4b1eafa2bdc1a9a3b7deffa/tinygrad/tensor.py