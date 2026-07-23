import cupy as cp

def cp_gpu_warmup():
    x = cp.ones((1000, 1000))
    y = x + x
    cp.cuda.Stream.null.synchronize()
