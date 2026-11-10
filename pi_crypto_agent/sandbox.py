"""Execute audit tests in a networkless, filesystem-isolated Bubblewrap sandbox."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def run_python_test(arguments, public):
    test=arguments.get('test_code')
    candidate=arguments.get('candidate_code')
    if not isinstance(test,str) or not test.strip() or len(test.encode())>16384:
        raise ValueError('test_code must contain1..16384 bytes')
    source=public['code'] if candidate is None else candidate
    if not isinstance(source,str) or len(source.encode())>65536:
        raise ValueError('Source code exceeds the64KiB limit')
    if Path(public['file']).suffix!='.py':
        return {'status':'unsupported_language','reason':'The current execution sandbox supports Python only; no false pass for other languages'}
    # Mount the interpreter's distribution, not the workspace or model directory.
    interpreter=Path(sys.executable).resolve()
    distribution=interpreter.parents[1]
    if not (distribution/'lib/python3.11').is_dir():
        raise RuntimeError('This sandbox is pinned to the project standalone Python3.11 distribution')
    mounts=[]
    for module in ['Crypto','cryptography','cffi','pycparser','_cffi_backend']:
        spec=importlib.util.find_spec(module)
        if spec and spec.origin:
            origin=Path(spec.origin)
            path=origin.parent if origin.name=='__init__.py' else origin
            mounts+=['--ro-bind',str(path),'/deps/'+path.name]
    bootstrap='''import resource,sys,runpy
resource.setrlimit(resource.RLIMIT_CPU,(5,5))
resource.setrlimit(resource.RLIMIT_AS,(512*1024*1024,512*1024*1024))
resource.setrlimit(resource.RLIMIT_FSIZE,(65536,65536))
resource.setrlimit(resource.RLIMIT_NPROC,(32,32))
resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
sys.path[:0]=['/deps','/workspace']
runpy.run_path('/workspace/test_audit.py',run_name='__main__')
'''
    with tempfile.TemporaryDirectory(prefix='crypto-audit-') as folder:
        work=Path(folder);(work/'audited_source.py').write_text(source)
        (work/'test_audit.py').write_text(test);(work/'bootstrap.py').write_text(bootstrap)
        command=['bwrap','--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL',
            '--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64',
            '--ro-bind',str(distribution),'/python','--dir','/deps',*mounts,
            '--ro-bind',str(work),'/workspace','--tmpfs','/tmp','--proc','/proc','--dev','/dev',
            '--clearenv','--setenv','PATH','/python/bin:/usr/bin','--setenv','HOME','/tmp',
            '--chdir','/workspace','/python/bin/python3.11','-I','/workspace/bootstrap.py']
        started=time.monotonic();timeout=False
        with (work/'stdout').open('wb') as out,(work/'stderr').open('wb') as err:
            child=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                timeout=True;os.killpg(child.pid,signal.SIGKILL);child.wait()
        stdout=(work/'stdout').read_bytes();stderr=(work/'stderr').read_bytes()
        return {'status':'timeout' if timeout else 'completed','exit_code':child.returncode,
            'test_passed':not timeout and child.returncode==0,'stdout':stdout[:12000].decode(errors='replace'),
            'stderr':stderr[:12000].decode(errors='replace'),'output_truncated':len(stdout)>12000 or len(stderr)>12000,
            'seconds':time.monotonic()-started,'source_kind':'candidate_fix' if candidate is not None else 'original',
            'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'test_sha256':hashlib.sha256(test.encode()).hexdigest(),
            'isolation':{'network':False,'host_workspace':False,'GPU_devices':False,'source_read_only':True,
                'CPU_seconds':5,'wall_seconds':10,'memory_MiB':512,'writable_directory':'ephemeral /tmp'},
            'interpretation':'A model-authored test passing is evidence for that test only, not proof of code security; private benchmark labels were not provided'}
