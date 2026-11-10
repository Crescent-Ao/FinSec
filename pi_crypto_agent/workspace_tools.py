"""Broad read/write/command tools inside a persistent per-case audit workspace."""
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

SESSIONS={}


class AuditWorkspace:
    def __init__(self,public):
        self.temp=tempfile.TemporaryDirectory(prefix='pi-audit-workspace-')
        self.root=Path(self.temp.name)
        (self.root/'audited_source.py').write_text(public['code'])
        self.original=public['code']

    def safe_path(self,name):
        if not isinstance(name,str) or not name or len(name)>256 or Path(name).is_absolute():
            raise ValueError('Use a relative workspace path')
        path=(self.root/name).resolve()
        if path!=self.root and self.root not in path.parents:raise ValueError('Path escapes the audit workspace')
        return path

    def command(self,command,wall=30):
        if not isinstance(command,str) or not command.strip() or len(command.encode())>16384:
            raise ValueError('Command must contain1..16384 bytes')
        if sum(p.lstat().st_size for p in self.root.rglob('*') if p.is_file() and not p.is_symlink())>8*1024*1024:
            raise ValueError('Audit workspace exceeds8MiB')
        distribution=Path(sys.executable).resolve().parents[1]
        if not (distribution/'lib/python3.11').is_dir():raise RuntimeError('Expected standalone Python3.11')
        mounts=[]
        for name in ['Crypto','cryptography','cffi','pycparser','_cffi_backend']:
            spec=importlib.util.find_spec(name)
            if spec and spec.origin:
                path=Path(spec.origin);path=path.parent if path.name=='__init__.py' else path
                mounts+=['--ro-bind',str(path),'/deps/'+path.name]
        with tempfile.TemporaryDirectory(prefix='pi-command-control-') as control:
            control=Path(control)
            bootstrap='''import os,resource
resource.setrlimit(resource.RLIMIT_CPU,(10,10))
resource.setrlimit(resource.RLIMIT_AS,(768*1024*1024,768*1024*1024))
resource.setrlimit(resource.RLIMIT_FSIZE,(131072,131072))
resource.setrlimit(resource.RLIMIT_NPROC,(64,64))
resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
os.execve('/usr/bin/bash',['bash','--noprofile','--norc','-c',open('/control/command').read()],dict(os.environ))
'''
            (control/'bootstrap.py').write_text(bootstrap);(control/'command').write_text(command)
            # Python subprocesses can use crypto dependencies via the isolated /deps.
            (control/'sitecustomize.py').write_text("import sys;sys.path.insert(0,'/deps')\n")
            cmd=['bwrap','--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL',
                '--ro-bind','/usr','/usr','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64',
                '--ro-bind',str(distribution),'/python','--dir','/deps',*mounts,
                '--bind',str(self.root),'/workspace','--ro-bind',str(control),'/control',
                '--tmpfs','/tmp','--proc','/proc','--dev','/dev','--clearenv',
                '--setenv','PATH','/python/bin:/usr/bin:/bin','--setenv','HOME','/tmp',
                '--setenv','PYTHONPATH','/control:/deps:/workspace','--chdir','/workspace',
                '/python/bin/python3.11','-I','/control/bootstrap.py']
            started=time.monotonic();timed_out=False
            with (control/'stdout').open('wb') as out,(control/'stderr').open('wb') as err:
                child=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True)
                try:child.wait(timeout=wall)
                except subprocess.TimeoutExpired:
                    timed_out=True;os.killpg(child.pid,signal.SIGKILL);child.wait()
            stdout=(control/'stdout').read_bytes();stderr=(control/'stderr').read_bytes()
            return {'exit_code':child.returncode,'timed_out':timed_out,'stdout':stdout[:16000].decode(errors='replace'),
                'stderr':stderr[:16000].decode(errors='replace'),'output_truncated':len(stdout)>16000 or len(stderr)>16000,
                'seconds':time.monotonic()-started,'network_enabled':False,'host_workspace_mounted':False,
                'workspace_persistent_between_tools':True,'command':command,
                'interpretation':'Execution evidence only. A passing model-authored test does not establish complete security.'}


def workspace_tool(name,args,public):
    key=hashlib.sha256((public.get('id','')+'\0'+public['file']+'\0'+public['code']).encode()).hexdigest()
    workspace=SESSIONS.setdefault(key,None)
    if workspace is None:workspace=SESSIONS[key]=AuditWorkspace(public)
    if name=='list_files':
        return {'files':[str(p.relative_to(workspace.root)) for p in workspace.root.rglob('*')][:200]}
    if name=='read_file':
        p=workspace.safe_path(args['path'])
        if p.stat().st_size>131072:raise ValueError('File exceeds read limit')
        return {'path':args['path'],'content':p.read_text()}
    if name=='write_file':
        content=args['content']
        if not isinstance(content,str) or len(content.encode())>65536:raise ValueError('File content exceeds64KiB')
        p=workspace.safe_path(args['path']);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
        return {'path':args['path'],'bytes':len(content.encode()),'written':True}
    if name=='replace_text':
        p=workspace.safe_path(args['path']);old=args['old'];new=args['new'];text=p.read_text()
        if not old or text.count(old)!=1:raise ValueError('Replacement must identify exactly one occurrence')
        changed=text.replace(old,new)
        if len(changed.encode())>65536:raise ValueError('Replacement exceeds64KiB')
        p.write_text(changed);return {'path':args['path'],'replacements':1,'written':True}
    if name=='run_command':return workspace.command(args['command'])
    if name=='workspace_info':return {'root':'/workspace','initial_source':'audited_source.py','write_access':True,
        'shell':True,'Python':True,'crypto_libraries':['Crypto','cryptography'],
        'network':False,'GPU':False,'host_files':False,'wall_timeout_seconds':30,'memory_MiB':768,
        'other_languages':'System compilers/interpreters may be used if present; missing dependencies are reported, not silently downloaded'}
    raise ValueError('Unknown workspace tool')
