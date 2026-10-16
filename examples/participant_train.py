"""Optional participant starter.

This file is deliberately participant-owned: replace it with the complete
training program you want to use. The current body delegates to the local PPO
reference only so a clean checkout has a runnable onboarding smoke test. The
reference model, optimizer, hyperparameters, and reward adapter are not part
of the SYNAPSE event contract.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_reference_baseline import main


if __name__ == "__main__":
    main()
