__all__ = ['parmap', 'parmapreduce']

import multiprocessing
from operator import add

def spawn_reduce(f, fun):
    def func(q_in,q_out):
        i,x = q_in.get()
        if i is None:
            q_out.put(None)
            return
        ret = f(x)
        while True:
            i,x = q_in.get()
            if i is None:
                break
            ret = fun(ret, f(x))
        q_out.put(ret)
    return func

def spawn(f):
    def func(q_in,q_out):
        while True:
            i,x = q_in.get()
            if i is None:
                break
            q_out.put((i,f(x)))
    return func

def parmap(f, X, nprocs = multiprocessing.cpu_count()):
    q_in   = multiprocessing.Queue(True)
    q_out  = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=spawn(f),args=(q_in,q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()
        proc

    [q_in.put((i,x)) for i,x in enumerate(X)]
    [q_in.put((None,None)) for _ in range(nprocs)]

    res = [q_out.get() for _ in range(len(X))]

    [p.join() for p in proc]
    res = [x for i,x in sorted(res)]
    return res

def parmapreduce(f, X, fun=add, nprocs = multiprocessing.cpu_count()):
    q_in   = multiprocessing.Queue(True)
    q_out  = multiprocessing.Queue()
    proc = [multiprocessing.Process(target=spawn_reduce(f, fun),args=(q_in,q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()
        proc

    [q_in.put((i,x)) for i,x in enumerate(X)]
    [q_in.put((None,None)) for _ in range(nprocs)]

    res = None
    for _ in range(nprocs):
        partial = q_out.get()
        if partial is not None:
            if res is None:
                res = partial
            else:
                res = fun(res, partial)

    [p.join() for p in proc]
    return res