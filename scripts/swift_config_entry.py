"""Typed Swift entry point, including nested kernel options without CLI coercion."""
import dataclasses
import json
import sys

from swift.cli.utils import try_use_single_device_mode
from swift.arguments import SftArguments
from swift.pipelines import sft_main


if __name__ == '__main__':
    try_use_single_device_mode()
    config = json.load(open(sys.argv[1]))
    unknown = set(config)-{f.name for f in dataclasses.fields(SftArguments) if f.init}
    if unknown: raise ValueError('Unknown SFT options: '+str(unknown))
    sft_main(SftArguments(**config))
