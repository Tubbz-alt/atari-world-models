#!/bin/bash
set -euo pipefail

find awm -name "*.py" -exec isort {} \;
