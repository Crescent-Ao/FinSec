"""Check a source release, or recompute an externally supplied experiment bundle."""
from pathlib import Path
import argparse,json

def verify_release(bundle_root=None):
    if bundle_root is None:
        from scripts.public_release_check import check
        result=check(Path(__file__).resolve().parent)
        print(json.dumps(result,ensure_ascii=False,indent=2));return result
    from submission_tools import offline_metrics
    offline_metrics.ROOT=Path(bundle_root).expanduser().resolve()
    if not (offline_metrics.ROOT/'evidence/expected_metrics.json').exists():
        raise FileNotFoundError('Provide an extracted experiment bundle containing evidence/expected_metrics.json')
    return offline_metrics.main()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle-root',type=Path)
    verify_release(p.parse_args().bundle_root)
