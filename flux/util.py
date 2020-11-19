import time

t0 = None

def tic():
    global t0
    t0 = time.perf_counter()

def toc():
    global t0
    t = time.perf_counter()
    return t - t0
