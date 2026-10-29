# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
"""``python -m openevolve_bedrock`` entry point."""

import sys

from openevolve_bedrock.cli import main

if __name__ == "__main__":
    sys.exit(main())
