# Steps


## Set up development environment

### User Java incubator module (in Kotlin gradle project)
1. https://foojay.io/today/how-to-run-the-java-incubator-module-from-the-command-line-and-intellij-idea/
2. https://stackoverflow.com/questions/70390734/how-can-i-use-the-incubating-vector-api-from-kotlin-in-a-gradle-build



## Implement Numpy 


### Numpy Basic Operations
Implement Numpy _Broadcasting_ and _Dot Product_ in pure Java/Kotlin, and use SIMD to accelerate computation.

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

### Implement backward
the original plan is to implement backward using closure same as the implementation in micrograd. however, after some investigation. I **suspect** that the closure may cause performance issues. so we transit to the tinygrad way   
* https://proandroiddev.com/kotlin-vs-java-the-inside-scoop-on-closures-ae9a8d6ddba5
* https://stackoverflow.com/questions/48140788/kotlin-higher-order-functions-costs
* https://magdamiu.medium.com/high-performance-with-idiomatic-kotlin-d52e099d0df0