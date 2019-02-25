# Benchmarks

I have implemented the algorithm in a few programming languages
just to see the performance on basic implementations. All math is
done in float32 precision.

All implementations and benchmarks are quick-n-dirty. The expected
runtime is O(n^2), results are in milliseconds. Reported numbers are
just the average of 20 runs without any outlier filtering or standard
deviation analysis.

Unlike matrix multiplication this test isn't actually that computationally
intensive. I guess the heaviest part are the transcendential functions `log`,
`exp`, `sin` and `cos`. The current formula has also some repeated calculations
as some calculated y-axis values are just the transpose of the x-axis values.
And the Gaussian blur `d` is multiplied element-wise with `k1` twice unless
the compiler is smart enough to eliminate this.


## System specs:

- Intel i7-6700K CPU @ 4.00GHz
- Nvidia GeForce 1060 GTX, 6 GB
- 64 GB RAM
- Ubuntu 18.04 LTS

## Runtime & library specs

- Octave: 4.2.2
- Python A: 3.6.8, GCC 7.3.0, NumPy 1.13.3, MKL, max 1 core(s)
- Python B: 3.6.8, GCC 7.3.0, NumPy 1.13.3, MKL, max 8 core(s)
- JCuda: org.jcuda/jcuda 10.0.0, uncomplicate/neanderthal 0.21.0, uncomplicate/clojurecuda 0.6.1, CUDA 10.0.130
- Neanderthal OpenCL (TBD)
- Julia (TBD)

## Results

```
              |     64 |    128 |    256 |     512 |     1024 |     2048 |      4096 |      8192 |      16384
--------------+--------+--------+--------+---------+----------+----------+-----------+-----------+------------
Octave    v1  |  1.042 |  3.175 | 12.760 |  50.867 |  200.467 |  892.580 |  4535.390 | 21890.426 | 124681.572
Python A  v1  |  0.703 |  1.731 |  6.747 |  27.859 |  117.866 |  551.634 |  2571.360 | 10378.900 |  42321.027
Python B  v1  |  0.575 |  0.884 |  4.731 |  15.229 |   68.154 |  328.142 |  1405.603 |  5226.997 |  21154.075
JCuda     v1a |  0.083 |  0.205 |  0.245 |   0.367 |   0.859  |    2.853 |    10.241 |    38.579 |    153.153
JCuda     v1b |  0.135 |  0.225 |  0.222 |   0.246 |   0.452  |    1.008 |     4.012 |    12.318 |     45.447
```

Notes:

- JCuda `v1b` measures only the kernel execution time, `v1a` includes also the time it takes to
transfer results from GPU to RAM. On larger images this step takes the majority of the time.
Note that `launch!` is synchronous so you need to call `synchronize!` to get correct results.

