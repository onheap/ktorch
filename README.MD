# Ktorch

<img src="docs/images/ktorch_logo.png" alt="Eval Logo" width="400" align="right">

## Intro
This project serves as an experimental playground for learning particular areas of knowledge using _learning by doing_ approach. The learning knowledge primary focus is on the following areas:
* Machine Learning
* PyTorch
* Kotlin
* New Java APIs (Vector API, Virtual Thread, Foreign Function & Memory API)
* Effective Java

## Steps

### Set up development environment

#### User Java incubator module (in Kotlin gradle project)
1. https://foojay.io/today/how-to-run-the-java-incubator-module-from-the-command-line-and-intellij-idea/
2. https://stackoverflow.com/questions/70390734/how-can-i-use-the-incubating-vector-api-from-kotlin-in-a-gradle-build


### Implement Numpy in Java

#### Numpy Basic Operations
Implement Numpy _Broadcasting_ and _Dot Product_ in pure Java/Kotlin, and use SIMD to accelerate computation.

#### References
* https://ajcr.net/stride-guide-part-1/
* https://numpy.org/doc/1.20/reference/internals.html
* https://numpy.org/doc/1.20/reference/arrays.ndarray.html

##### Broadcasting
* https://numpy.org/doc/stable/user/basics.broadcasting.html

##### Dot Product
* https://www.reddit.com/r/java/comments/17pkcgx/how_fast_can_we_do_matrix_multiplications_in_pure/
* https://www.elastic.co/blog/accelerating-vector-search-simd-instructions
* Whether we need to use _Virtual Thread_ to parallelize Matrix Multiplication? Answer is NO. Virtual Thread is more suitable for I/O bound tasks. And Matrix Multiplication is a CPU bound task.  [ref](https://www.reddit.com/r/java/comments/16mkm4v/efficiency_of_java_21_virtual_threads_compared_to/)
* https://github.com/lessthanoptimal/VectorPerformance/blob/master/src/main/java/benchmark/MatrixMultiplication.java
* https://www.baeldung.com/jvm-tiered-compilation

##### Performance testing (JMH)
Principle: make matrix multiplication as fast as possible, and other operations more elegant.
* https://github.com/melix/jmh-gradle-plugin
* https://github.com/openjdk/jdk/pull/15338


### Implement Simple NN

#### Implement backward
https://github.com/tinygrad/tinygrad/blob/91a352a8e2697828a4b1eafa2bdc1a9a3b7deffa/tinygrad/tensor.py

the original plan is to implement backward using closure same as the implementation in micrograd. however, after some investigation. I **suspect** that the closure may cause performance issues. so we transit to the tinygrad way
* https://proandroiddev.com/kotlin-vs-java-the-inside-scoop-on-closures-ae9a8d6ddba5
* https://stackoverflow.com/questions/48140788/kotlin-higher-order-functions-costs
* https://magdamiu.medium.com/high-performance-with-idiomatic-kotlin-d52e099d0df0

#### Implement MNIST
```shell
./gradlew run --args=mnist
```

https://github.com/tinygrad/tinygrad/blob/91a352a8e2697828a4b1eafa2bdc1a9a3b7deffa/test/mnist.py

* Predict: input.dot(l1).relu().dot(l2).logSoftmax()
* Loss: NLL loss or [CrossEntropyLoss](https://stackoverflow.com/questions/65192475/pytorch-logsoftmax-vs-softmax-for-crossentropyloss)

##### Tool
* Print image in terminal
  * https://stackoverflow.com/questions/5762491/how-to-print-color-in-console-using-system-out-println
  * https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797
  * https://github.com/Nellousan/px2ansi/blob/main/px2ansi.py
  ```shell
  ████████████████████████████
  ████████████████████████████
  █████████████▀▀▀▀███████████
  █████████████▀  ▄▄ ▀████████
  ████████▀  ▄▄▄▄  █  ████████
  ████████  ██████▀ ▄█████████
  ████████▄▄ ▀▀█▀ ▄███████████
  ████████████   █████████████
  ███████████  █▄ ▀███████████
  ██████████  ████▄ ██████████
  ██████████ ██████ ██████████
  ██████████ ▀██▀▀  ██████████
  ███████████▄▄▄▄▄████████████
  ████████████████████████████
  ```

* Progress bar:
  * https://medium.com/javarevisited/how-to-display-progressbar-on-the-standard-console-using-java-18f01d52b30e
  * https://github.com/ctongfei/progressbar/blob/main/src/main/java/me/tongfei/progressbar/ProgressBarStyle.java

  ```shell
  Training 1950/2500  78%: │███████████████████▌     │ 4.316723s loss 0.041252743, accuracy 0.90625
  ```



# TODO List
* Get sub NDArray by indices
* Make matmul applicable for 1-d arrays, ref [implementation in mlx](https://github.com/ml-explore/mlx/blob/026ef9aae4ba33743e34bff554529a2b6ff7ab54/mlx/ops.cpp#L1934-L1941)
* Unroll matmul, ref [llama2.java](https://github.com/mukel/llama2.java/blob/80f5a858c8c44bbf4358cc3f0c3a281e9b7c03bc/Llama2.java#L294)
* Benchmark with flame graph & memory usage


# Reference 

## Articles
* https://github.com/pytorch/pytorch
* https://pytorch.org/docs/stable/tensors.html
* https://github.com/karpathy/nn-zero-to-hero
* https://github.com/mikex86/scicore
* http://blog.ezyang.com/2019/05/pytorch-internals/
* https://blog.paperspace.com/pytorch-101-advanced/
* https://blog.christianperone.com/2019/02/pydata-montreal-slides-for-the-talk-pytorch-under-the-hood/
* https://blog.christianperone.com/2018/03/pytorch-internal-architecture-tour/

## Projects
* https://github.com/tinygrad/tinygrad/ ([early stage commit](https://github.com/tinygrad/tinygrad/tree/91a352a8e2697828a4b1eafa2bdc1a9a3b7deffa))
* https://github.com/karpathy/micrograd
* https://github.com/mikex86/scicore
* https://github.com/deepjavalibrary/djl
* https://github.com/lessthanoptimal/ejml
* https://github.com/ml-explore/mlx
* https://github.com/mukel/llama2.java
* https://github.com/padreati/rapaio
* https://github.com/Kotlin/multik
