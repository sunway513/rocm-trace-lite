#!/usr/bin/env python3
"""Emit the original bundle's library directories, including split ROCm SDK wheels."""
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
manifest = json.loads((root / 'manifest.json').read_text())
print(':'.join(str(root / relative) for relative in manifest['library_directories']))
