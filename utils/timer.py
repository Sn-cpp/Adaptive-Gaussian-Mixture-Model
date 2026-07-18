from time import perf_counter
import cupy as cp

def cpu_timer(func, *args, **kwargs):
    """
    Wrapper function for measuring execution time of a function on CPU ONLY

    Parameter:
        + `func`: To-be-measured function
        + `args`, `kwargs`: Function arguments
    """
    
    start = perf_counter()
    ret = func(*args, **kwargs)
    end = perf_counter()

    return ret, end-start

def gpu_timer(func, *args, **kwargs):
    """
    Wrapper function for measuring execution time of a function on GPU via CuPy

    Parameter:
        + `func`: To-be-measured function
        + `args`, `kwargs`: Function arguments
    """

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    ret = func(*args, **kwargs)
    end.record()
    end.synchronize() 

    return ret, cp.cuda.get_elapsed_time(start, end)



