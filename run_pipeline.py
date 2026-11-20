"""Entry point so the pipeline can be started as `python run_pipeline.py --stage all`.

All the logic lives in pipeline/run_pipeline.py; this file only forwards to it.
"""

import sys

from pipeline.run_pipeline import main

if __name__ == "__main__":
    sys.exit(main())
