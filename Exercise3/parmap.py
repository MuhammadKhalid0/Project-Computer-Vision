import progressbar
import multiprocessing
from multiprocessing.pool import ThreadPool

def spawn(f):
    def fun(q_in, q_out):
        while True:
            i,x = q_in.get()
            if i is None:
                break
            q_out.put((i, f(x)))

    return fun

def parmap(f, iterable, nprocs=multiprocessing.cpu_count(), 
           show_progress=False, size=None):
    """
    @param f
    function to be applied to the items in iterable
    @param iterable
    ...
    @param nprocs
    number of processes
    @param show_progress
    True <-> show a progress bar
    @param size
    number of items in iterable.
    If show_progress == True and size is None and iterable is not already a
    list, it is converted to a list first. This could be bad for generators!
    (If size is not needed right away for the progress bar, all input items
    are enqueued before reading the results from the output queue.)
    TLDR: If you know it, tell us the size of your iterable.
    """
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    progress = None
    if show_progress:
        if not isinstance(iterable, list):
            iterable = list(iterable)
        size = len(iterable)

        widgets = [ progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA() ]
        progress = progressbar.ProgressBar(widgets=widgets, maxval=size)

    proc = [ multiprocessing.Process(target=spawn(f), args=(q_in, q_out)) for _ in range(nprocs) ]

    for p in proc:
        p.daemon = True
        p.start()

    if progress is not None:
        progress.start()


    def enqueue():
        s = 0
        for i, x in enumerate(iterable):
            q_in.put((i,x))
            s += 1

        for _ in range(nprocs):
            q_in.put((None,None))

        return s

    pool = ThreadPool(processes=1)
    async_size = pool.apply_async(enqueue)

    if size is None:
        # this is the old behavior
        size = async_size.get()

    res = []
    progress_value = 0
    for _ in range(size):
        r = q_out.get()
        res.append(r)

        # we could: insert sorted, yield all results we have so far

        if progress is not None:
            progress_value += 1
            progress.update(progress_value)

    del pool
    for p in proc:
        p.join()

    if progress is not None:
        progress.finish()

    return [ x for _, x in sorted(res) ]

