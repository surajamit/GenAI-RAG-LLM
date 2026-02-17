import time
from contextlib import contextmanager


@contextmanager
def timer(name: str):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"[LATENCY] {name}: {(end - start)*1000:.2f} ms")


from utils.profiler import timer

with timer("embedding"):
    embeddings = model.encode(texts)