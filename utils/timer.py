from time import perf_counter
import cupy as cp

def cpu_timer(func, args = []):
    """
    Wrapper function for measuring execution time of a function on CPU ONLY

    Parameter:
        + `func`: To-be-measured function
        + `args`: Function arguments
    """
    
    if isinstance(args, dict):
        start = perf_counter()
        ret = func(**args)
        end = perf_counter()
    else:
        start = perf_counter()
        ret = func(*args)
        end = perf_counter()
    
    return ret, end-start

def gpu_timer(func, args):
    """
    Wrapper function for measuring execution time of a function on GPU via CuPy

    Parameter:
        + `func`: To-be-measured function
        + `args`: Function arguments
    """

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    if isinstance(args, dict):
        start.record()
        ret = func(**args)
        end.record()
        end.synchronize()
    else:
        start.record()
        ret = func(*args)
        end.record()
        end.synchronize()

    return ret, cp.cuda.get_elapsed_time(start, end)



