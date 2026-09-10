"""Four-device model parallelism, without splitting the source timeline.

One serial worker owns each model stage. In particular, VballNet keeps its
rolling seq9 buffer and radius history across every 90-frame decode chunk.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import re
import threading
import time


ROLES = ('state', 'action', 'person', 'tracking')


def parse_devices(value):
    if value is None:
        return None
    values = value.split(',') if isinstance(value, str) else list(value)
    devices = [str(d).strip() for d in values]
    if len(devices) != 4 or any(not re.fullmatch(r'cuda:[0-9]+', d) for d in devices):
        raise ValueError('--devices requires four CUDA devices, e.g. cuda:0,cuda:1,cuda:2,cuda:3')
    devices = [f'cuda:{int(d.split(":")[1])}' for d in devices]
    if len(set(devices)) != 4:
        raise ValueError('--devices requires four distinct CUDA devices')
    return devices


class GPUStages:
    def __init__(self, devices, *, bind_cuda=True):
        self.devices = dict(zip(ROLES, parse_devices(devices)))
        self.bind_cuda = bind_cuda
        self.pools = {}
        self.lock = threading.Lock()
        self.active = self.max_active = 0
        self.stats = {role: {'device': device, 'tasks': 0, 'wall_sec': 0.}
                      for role, device in self.devices.items()}

    def __enter__(self):
        self.pools = {role: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'gpu-{role}')
                      for role in ROLES}
        return self

    def submit(self, role, function, *args):
        def execute():
            context = nullcontext()
            if self.bind_cuda:
                import torch
                context = torch.cuda.device(self.devices[role])
            with context:
                start = time.perf_counter()
                with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                try:
                    return function(*args)
                finally:
                    with self.lock:
                        self.active -= 1
                        self.stats[role]['tasks'] += 1
                        self.stats[role]['wall_sec'] += time.perf_counter()-start
        return self.pools[role].submit(execute)

    def __exit__(self, *exc):
        for pool in self.pools.values():
            pool.shutdown(wait=True, cancel_futures=True)

    def report(self):
        return {'strategy': 'four-device-model-parallel-pts90-v1',
                'role_devices': self.devices, 'max_concurrent_stage_tasks': self.max_active,
                'stages': {role: {**stats, 'wall_sec': round(stats['wall_sec'], 3)}
                           for role, stats in self.stats.items()},
                'note': 'Stage wall times overlap; they are not GPU kernel timings. Tracking and auxiliary ball share one device.'}
