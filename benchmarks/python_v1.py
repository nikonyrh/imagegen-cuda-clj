# docker build -t python3 python
# docker run --rm -it -e N_CORES=4 --volume "$PWD:/src" python3 src/python_v1.py

import os
import ctypes
import time
import numpy as np

n_cores = os.environ.get('N_CORES')
if n_cores:
    n_cores = int(n_cores)
    print('n_cores = %d\n' % n_cores)
    
    try:
        import mkl
        mkl.set_num_threads(n_cores)
    except:
        import ctypes
        ctypes.CDLL('libmkl_rt.so').mkl_set_num_threads(ctypes.byref(ctypes.c_int(n_cores)))


def to_log(i, res, eps=1e-6):
    f = i / (res - 1) * (1 - 2 * eps) + eps
    return np.log(f / (1.0 - f))

def to_normal(i):
    return 1.0 / (1.0 + np.exp(-i))


for res in [2**i for i in range(6, 14 + 1)]:
    pi2 = np.pi * 2
    n_iter = 20
    x, y = np.meshgrid(*[range(res) for _ in range(2)])
    
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    im = np.zeros((res, res, 3)).astype(np.uint8)
    
    t0 = time.time()
    
    for _ in range(n_iter):
        subres = res // 4
        i = to_log(np.bitwise_and(x.astype(int), subres - 1), subres).astype(np.float32)
        j = to_log(np.bitwise_and(y.astype(int), subres - 1), subres).astype(np.float32)
        
        d = np.exp(-0.35 * (i**2 + j**2))
        
        k1 = 1.0 + 2.0 * (x // subres)
        k2 = 0.5 + 0.5 * (y // subres)
        
        im[:,:,0] = (to_normal(i + k1 * d * np.sin(j * k2 * pi2)) * 255).round()
        im[:,:,1] = (to_normal(j + k1 * d * np.cos(i * k2 * pi2)) * 255).round()
        im[:,:,2] = (to_normal((i + j) * 0.5) * 255).round()
    
    t1 = time.time()
    print('Done in %10.3f ms (%4d x %4d)' % (1e3 * (t1 - t0) / n_iter, res, res))


if False:
    import matplotlib.pyplot as plt
    plt.imshow(im)
    plt.tight_layout()
