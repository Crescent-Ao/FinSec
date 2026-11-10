"""Local verified rewards. Private task answers stay outside the model prompt."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stage2_verifiers import verify
from swift.rewards import ORM, orms


class FinancialSecurityReward(ORM):
    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        self.tasks = {}
        for filename in os.environ['STAGE2_REWARD_TASKS'].split(os.pathsep):
            for line in Path(filename).read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.tasks[row['id']] = row

    def __call__(self, completions, solution, **kwargs):
        rewards = []
        for completion, identity in zip(completions, solution):
            task = self.tasks[identity]
            result = verify(task, {'completion': completion})
            rewards.append(result['reward'])
        return rewards


orms['financial_security_verified'] = FinancialSecurityReward
