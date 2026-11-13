"""Source-only publication checks. Reports file names, never credential values."""
from pathlib import Path
import ast,json,re,subprocess

FORBIDDEN_PARTS={'models','datasets','data-distill','evaluation_data','evaluation_models','outputs','runs','runtime','evidence','reports','documents','exports','node_modules','.secrets','.venv'}
FORBIDDEN_SUFFIX={'.safetensors','.pt','.pth','.ckpt','.bin','.gguf','.onnx','.pkl','.jsonl','.parquet','.arrow','.bundle','.zip','.gz','.pdf','.docx'}
PATTERNS={
 'API credential':re.compile(r'\b(?:sk-[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
 'private key':re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
 'ModelScope credential':re.compile(r'\bms-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'),
 'inline SSH password':re.compile(r'sshpass\s+-p\s+[^\s]+'),
}

def check(root):
    root=Path(root).resolve();tracked=subprocess.run(['git','ls-files','-z'],cwd=root,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    names=[n for n in tracked.stdout.decode().split('\0') if n] if tracked.returncode==0 else []
    if not names:
        names=[str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and not any(x in {'.git','__pycache__','node_modules','.venv','.cache'} for x in p.relative_to(root).parts)]
    issues=[];python_files=0;total=0
    for name in names:
        path=Path(name);p=root/path
        if any(part in FORBIDDEN_PARTS for part in path.parts) or p.suffix.lower() in FORBIDDEN_SUFFIX:issues.append((name,'non-source artifact'))
        if p.name=='.env' or p.name.startswith('.env.') and p.name!='.env.example':issues.append((name,'local environment file'))
        total+=p.stat().st_size
        if p.stat().st_size>2_000_000:issues.append((name,'unexpected large file'))
        if p.suffix in {'.py','.mjs','.js','.json','.yaml','.yml','.md','.txt','.sh','.html','.in'} or p.name=='.env.example':
            text=p.read_text()
            for kind,pattern in PATTERNS.items():
                if pattern.search(text):issues.append((name,kind))
            if p.suffix=='.py':ast.parse(text,filename=name);python_files+=1
    if issues:raise ValueError('Publication check failed: '+json.dumps(issues,ensure_ascii=False))
    return {'source_files':len(names),'python_files_parsed':python_files,'bytes':total,'weights_in_git':False,
            'private_data_in_git':False,'credential_scan':'passed','syntax_check':'passed'}

if __name__=='__main__':print(json.dumps(check(Path(__file__).resolve().parents[1]),ensure_ascii=False,indent=2))
