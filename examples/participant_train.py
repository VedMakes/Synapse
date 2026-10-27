"""Optional participant starter.

This file is deliberately participant-owned: you can replace it with the
training program you want to use. The current body delegates to the local PPO
reference.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_reference_baseline import main

if __name__ == "__main__":
    main()
