"""Wall time and sampled GPU use; distinguish process allocation from whole-device use."""
import json
import os
import subprocess
import threading
import time


class UsageMonitor:
    def __init__(self, device):
        self.devices = list(device) if isinstance(device, (list, tuple)) else [device]
        self.device = self.devices[0]
        self.samples = []
        self.error = None
        self.stop_event = threading.Event()
        self.thread = None

    def __enter__(self):
        self.started = time.perf_counter()
        if self.device.startswith('cuda'):
            import torch
            # is_available() does not initialize the caching allocator. Resetting
            # stats before lazy initialization raises "Invalid device argument".
            torch.cuda.init()
            for device in self.devices:
                torch.cuda.reset_peak_memory_stats(device)
            self.thread = threading.Thread(target=self.poll, daemon=True)
            self.thread.start()
        return self

    def poll(self):
        while not self.stop_event.is_set():
            try:
                value = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'], text=True, timeout=4)
                self.samples.append({'elapsed_sec': time.perf_counter()-self.started,
                    'devices': [line.split(', ') for line in value.strip().splitlines()]})
            except (OSError, subprocess.SubprocessError) as exc:
                self.error = type(exc).__name__
                break
            self.stop_event.wait(.5)

    def __exit__(self, *exc):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        self.elapsed = time.perf_counter()-self.started

    def report(self):
        result = {'device': self.device, 'devices': self.devices, 'pid': os.getpid(),
                  'wall_sec': round(self.elapsed,3),
                  'gpu_samples': self.samples, 'gpu_sampling_error': self.error,
                  'memory_note': 'nvidia-smi samples cover whole devices, including other processes; torch peaks exclude ORT allocations.'}
        if self.device.startswith('cuda'):
            import torch
            result['torch_devices'] = {device: {
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
                'name': torch.cuda.get_device_name(device)} for device in self.devices}
            result.update(torch_cuda_version=torch.version.cuda)
            if len(self.devices) == 1:
                result.update(torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                          torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
                          gpu_name=torch.cuda.get_device_name(self.device))
        return result
