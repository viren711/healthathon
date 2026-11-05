"""Rewrite sarvam_api_usage.txt from logs/sarvam_usage.jsonl.

    python scripts/12_sarvam_usage.py

The app already rewrites this file after every Sarvam call; this is for
rebuilding it by hand. Costs are the app's own estimates from the published
prices in config/voice.yaml. Sarvam has no balance endpoint, so "credits"
means rupees spent; the Sarvam dashboard shows what is left.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rag.sarvam import load_voice_config, write_usage_report  # noqa: E402

cfg = load_voice_config()
text = write_usage_report(ROOT / cfg["paths"]["usage_log"], ROOT / cfg["paths"]["usage_report"], cfg)
print("\n".join(text.splitlines()[:16]))
