"""Reproducible bias generation and isolated imcombine measurements.

Run the worker through ``measure_combination`` rather than importing a notebook
function into a spawned process. Timing and memory always use separate workers.
"""

import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from tempfile import TemporaryDirectory
from time import perf_counter
import tracemalloc

import numpy as np


def make_bias_frames(*, seed, shape, n_frames, bias_level, read_noise,
                     positive_spike_rate, positive_spike_range,
                     negative_spike_rate, negative_spike_range):
    """Create the same seeded uint16 bias sequence for plots and benchmarks."""
    from astropop.framedata import FrameData

    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n_frames):
        pixels = rng.normal(bias_level, read_noise, shape)
        for sign, rate, bounds in [(1, positive_spike_rate, positive_spike_range),
                                   (-1, negative_spike_rate, negative_spike_range)]:
            count = int(rate*np.prod(shape))
            indices = rng.choice(pixels.size, count, replace=False)
            pixels.ravel()[indices] += sign*rng.uniform(*bounds, count)
        if pixels.min() < 0 or pixels.max() > np.iinfo('uint16').max:
            raise ValueError('Simulation produced pixels outside uint16 range')
        frames.append(FrameData(pixels.astype('uint16'), dtype='uint16', unit='adu'))
    return frames


def measure_combination(simulation, options, dtype, *, memory_limit,
                        package_root, repeats=2, threads=1, timeout=600):
    """Return timing and memory metrics, excluding imports/input generation.

    The timing worker runs without tracing. A separate memory worker combines
    once, after generating inputs, with tracemalloc and a 5 ms RSS sampler.
    Timings are wall seconds; memory fields contain bytes. RSS sampling can miss
    short peaks; traced allocations cover NumPy buffers if its allocator hook
    is available. Input data/flags are included in baseline RSS, not tracing.
    Workers inherit the caller's FrameData uncertainty/flag configuration.
    Disabled flag arrays contribute zero bytes to ``input_flags``.
    """
    from astropop.config import AstropopConfig as conf

    configuration = {name: bool(getattr(conf, name)) for name in
                     ('FRAMEDATA_DISABLE_FLAGS', 'FRAMEDATA_DISABLE_UNCERTAINTY')}
    if repeats < 1 or threads < 1:
        raise ValueError('repeats and threads must be positive')
    job = dict(simulation=simulation, options=options, dtype=np.dtype(dtype).name,
               memory_limit=memory_limit, package_root=str(package_root), repeats=repeats,
               configuration=configuration)
    env = os.environ.copy()
    for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS'):
        env[variable] = str(threads)
    measurements = {}
    for mode in ('timing', 'memory'):
        with TemporaryDirectory(prefix='imcombine-benchmark-') as cache:
            Path(cache, 'astropy').mkdir()
            worker_env = dict(env, XDG_CACHE_HOME=cache)
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), json.dumps(dict(job, mode=mode))],
                cwd=Path(__file__).resolve().parents[1], env=worker_env,
                capture_output=True, text=True, timeout=timeout)
        if process.returncode:
            raise RuntimeError(f'{mode} benchmark failed:\n{process.stdout}\n{process.stderr}')
        result = json.loads(process.stdout)
        measurements.update(result)
    return measurements


def _worker(job):
    # Resolve the same package as the calling kernel, before importing Astropop.
    sys.path.insert(0, job['package_root'])
    import psutil
    from astropop.config import AstropopConfig as conf
    from astropop.image.imcombine import imcombine
    from astropop.logger import logger

    import logging
    logger.setLevel(logging.WARNING)
    for name, disabled in job['configuration'].items():
        setattr(conf, name, disabled)
    frames = make_bias_frames(**job['simulation'])
    kwargs = dict(job['options'], dtype=job['dtype'], memory_limit=job['memory_limit'])
    gc.collect()
    if job['mode'] == 'timing':
        durations = []
        for _ in range(job['repeats']):
            started = perf_counter()
            result = imcombine(frames, **kwargs)
            durations.append(perf_counter()-started)
            del result
            gc.collect()
        return dict(seconds=durations, configuration=job['configuration'])

    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    peak_rss = [baseline_rss]
    stop = threading.Event()

    def sample_rss():
        while not stop.wait(0.005):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    tracemalloc.start()
    # Verify that NumPy's buffers are visible before interpreting the peak.
    probe = np.empty(2**20, dtype=np.uint8)
    numpy_tracked = tracemalloc.get_traced_memory()[1] >= probe.nbytes
    del probe
    traced_baseline = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    try:
        result = imcombine(frames, **kwargs)
        peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
        traced_peak = tracemalloc.get_traced_memory()[1]-traced_baseline
    finally:
        tracemalloc.stop()
        stop.set()
        sampler.join()
    if not numpy_tracked:
        raise RuntimeError('NumPy buffers are not visible to tracemalloc on this build')
    return dict(baseline_rss=baseline_rss, peak_rss=peak_rss[0],
                extra_rss=max(0, peak_rss[0]-baseline_rss), traced_peak=traced_peak,
                input_data=sum(f.data.nbytes for f in frames),
                input_flags=sum(flags.nbytes for flags in (f.flags for f in frames)
                                if flags is not None),
                output_flags=result.flags is not None,
                output_uncertainty=result.uncertainty is not None,
                configuration=job['configuration'])


if __name__ == '__main__':
    print(json.dumps(_worker(json.loads(sys.argv[1]))))
