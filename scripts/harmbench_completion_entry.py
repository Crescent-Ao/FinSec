"""Run upstream answer generation with a recorded, bounded starting batch."""
import argparse
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fineval_pipeline.core import digest,write_json
from fineval_pipeline.harmbench_completions import capped_batch_finder


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--upstream-script',type=Path,required=True)
    parser.add_argument('--batch-cap',type=int,required=True)
    args,remaining=parser.parse_known_args()
    if args.batch_cap<1:raise ValueError('Completion batch cap must be positive')
    script=args.upstream_script.absolute()
    sys.path.insert(0,str(script.parent))
    sys.argv=[str(script),*remaining]
    namespace=runpy.run_path(str(script),run_name='harmbench_completion_entry')
    scope=namespace['main'].__globals__
    scope['find_executable_batch_size']=capped_batch_finder(scope['find_executable_batch_size'],args.batch_cap)
    saved=Path(remaining[remaining.index('--save_path')+1])
    write_json(saved.with_name(saved.stem+'_execution.json'),{
        'upstream_script_sha256':digest(script),'entry_sha256':digest(__file__),
        'batch_adapter_sha256':digest(ROOT/'fineval_pipeline/harmbench_completions.py'),
        'starting_batch_cap':args.batch_cap,'arguments':remaining,
        'change':'cap initial HF generation batch; retain upstream OOM backoff and generation parameters'})
    namespace['main']()


if __name__=='__main__':main()
