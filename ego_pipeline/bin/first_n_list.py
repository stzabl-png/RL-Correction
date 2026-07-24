#!/usr/bin/env python
"""Print the first N video ids discovered under Reconstruct_and_Retarget/Data (one per line).

Shared by label.sh and reconstruct.sh so labeling and reconstruction target the
SAME N takes. Discovery is deterministic, so independent calls agree.

  python first_n_list.py [N]            # default 10
"""
import sys
from pathlib import Path

RECON = Path("/home/lyh/Project/HumanVideo2RobotData/recon_pipeline")
DATA_ROOT = Path("/home/lyh/Project/Reconstruct_and_Retarget/Data/HOI4D")  # Data/ 现仅 HOI4D；新增数据集时在此扩展
sys.path.insert(0, str(RECON))

from _common.dataset import discover_videos  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
jobs = discover_videos("hoi4d", dataset_root=DATA_ROOT, limit=n)
for j in jobs:
    print(j.video_id)
