"""
OSM end‑to‑end job


Steps
1)

Environment
- MINIO

"""
import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from typing import Iterable, Optional, Tuple


from research_code.ops.osm_store import *