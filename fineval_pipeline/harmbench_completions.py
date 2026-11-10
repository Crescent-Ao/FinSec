"""Bound completion batch discovery without changing the frozen search runtime."""
from functools import partial
from pathlib import Path

from .core import ROOT


def capped_batch_finder(finder, cap):
    if cap < 1:
        raise ValueError('Completion batch cap must be positive')

    def bounded(function=None, starting_batch_size=128, **kwargs):
        if function is None:
            return partial(bounded, starting_batch_size=starting_batch_size, **kwargs)

        def logged(batch_size, *args, **options):
            print('HB_COMPLETION_BATCH_SIZE', batch_size, flush=True)
            return function(batch_size, *args, **options)
        return finder(logged, starting_batch_size=min(cap, starting_batch_size), **kwargs)
    return bounded


def completion_command(cfg, runtime, command):
    if not command or command[0] != 'generate_completions.py':
        return command
    cap=int(cfg['harmbench'].get('completion_batch_cap',32))
    if cap < 1:raise ValueError('Completion batch cap must be positive')
    return [str(ROOT/'scripts/harmbench_completion_entry.py'),
            '--upstream-script',str(Path(runtime)/command[0]),'--batch-cap',str(cap),*command[1:]]


def install():
    from . import harmbench_upstream
    original=harmbench_upstream.run_command
    if getattr(original,'bounded_completions',False):return

    def run_command(cfg,runtime,command,log):
        return original(cfg,runtime,completion_command(cfg,runtime,command),log)
    run_command.bounded_completions=True
    harmbench_upstream.run_command=run_command
