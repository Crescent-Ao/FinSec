"""One BF16 evaluation representation shared by all four evaluation tasks."""
import json

from .core import Backend, digest, model_identity, model_spec, resolve, write_json


def export(cfg,name):
    spec=model_spec(cfg,name,use_export=False)
    if spec.get('precision','bf16')!='bf16':
        raise ValueError('Export requires BF16 source loading; never merge into a quantized base')
    destination=resolve(cfg['models'][name]['evaluation_export'])
    identity=model_identity(spec)
    if (destination/'export_manifest.json').exists():
        saved=json.loads((destination/'export_manifest.json').read_text())
        if saved['source_identity']!=identity:raise ValueError('Existing export belongs to another model')
        for row in saved['files']:
            if digest(destination/row['file'])!=row['sha256']:raise ValueError('Export file changed')
        return saved
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Incomplete or untracked export directory; use a new evaluation_export')
    destination.mkdir(parents=True,exist_ok=True)
    backend=Backend(spec)
    model=backend.model.merge_and_unload(safe_merge=True) if spec.get('adapter') else backend.model
    model.save_pretrained(destination,safe_serialization=True,max_shard_size='5GB')
    backend.tokenizer.save_pretrained(destination)
    report={'model':name,'source_identity':identity,'transformation':'BF16 merged PEFT export' if spec.get('adapter') else 'BF16 copy',
            'files':[{'file':p.name,'sha256':digest(p)} for p in sorted(destination.iterdir()) if p.is_file()]}
    write_json(destination/'export_manifest.json',report)
    return report
