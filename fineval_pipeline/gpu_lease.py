"""Serialize this project's A100 training and final evaluation processes."""
import fcntl
import os
import time

from .core import ROOT,write_json


def acquire_a100_lease(owner,status_path):
    path=ROOT/'.cache/a100_training_evaluation.lock';path.parent.mkdir(exist_ok=True)
    handle=path.open('a')
    while True:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            # A MIA worker may have been explicitly co-located beside the last
            # PAIR target. It keeps its supplementary slot until model unload;
            # the next exclusive classifier/training job must wait for it too.
            supplementary=ROOT/'.cache/A100_MIA_alongside_GRPO_PAIR.lock'
            if supplementary.exists():
                guard=supplementary.open('a')
                fcntl.flock(guard,fcntl.LOCK_EX)
                guard.close()
            write_json(status_path,{'state':'acquired','owner':owner,'pid':os.getpid(),'time':time.time()})
            return handle
        except BlockingIOError:
            write_json(status_path,{'state':'waiting','owner':owner,'pid':os.getpid(),'time':time.time()})
            time.sleep(15)


def acquire_local_gpu_lease(gpu, owner, status_path):
    """Keep a multi-step local evaluation on one card until all its stages end."""
    if type(gpu) is not int or gpu < 0:
        raise ValueError('A nonnegative local GPU index is required')
    path = ROOT / '.cache' / ('local_gpu_' + str(gpu) + '.lock')
    path.parent.mkdir(exist_ok=True)
    handle = path.open('a')
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            write_json(status_path, {'state': 'acquired', 'owner': owner, 'gpu': gpu,
                                    'pid': os.getpid(), 'time': time.time()})
            return handle
        except BlockingIOError:
            write_json(status_path, {'state': 'waiting', 'owner': owner, 'gpu': gpu,
                                    'pid': os.getpid(), 'time': time.time()})
            time.sleep(15)


def try_acquire_local_gpu_lease(gpu, owner, status_path):
    """Take one GPU immediately or close its handle and let a scheduler try another."""
    if type(gpu) is not int or gpu < 0:
        raise ValueError('A nonnegative local GPU index is required')
    path = ROOT / '.cache' / ('local_gpu_' + str(gpu) + '.lock')
    path.parent.mkdir(exist_ok=True)
    handle = path.open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    write_json(status_path, {'state': 'acquired', 'owner': owner, 'gpu': gpu,
                            'pid': os.getpid(), 'time': time.time()})
    return handle
