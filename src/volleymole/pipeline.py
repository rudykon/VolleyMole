"""Bounded, ordered cross-batch inference; one owner for each temporal model."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager


@contextmanager
def prefetch_batches(batches, timings):
    """At most one decoded batch ahead. Exceptions propagate to the consumer."""
    iterator = iter(batches)
    sentinel = object()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='decode') as pool:
        pending = pool.submit(next, iterator, sentinel)

        def consume():
            nonlocal pending
            while True:
                with timings.measure('decode_consumer_wait'):
                    batch = pending.result()
                if batch is sentinel:
                    return
                pending = pool.submit(next, iterator, sentinel)
                yield batch
        try:
            yield consume()
        finally:
            pending.cancel()
            # Join the owner before closing the generator, including failures.
            pool.shutdown(wait=True, cancel_futures=True)
            if hasattr(iterator, 'close'):
                iterator.close()


def ordered_inference(batches, submit, resolve, depth, timings):
    if depth < 1 or depth > 4:
        raise ValueError('pipeline depth must be 1..4')
    pending = deque()
    for batch in batches:
        pending.append((batch, submit(batch)))
        if len(pending) >= depth:
            packets, job = pending.popleft()
            with timings.measure('inference_consumer_wait'):
                result = resolve(job)
            yield packets, result
    while pending:
        packets, job = pending.popleft()
        with timings.measure('inference_consumer_wait'):
            result = resolve(job)
        yield packets, result
