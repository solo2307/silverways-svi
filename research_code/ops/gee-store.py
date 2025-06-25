"""Extract DATA from Google Earth Engine"""
import ee
from ee import ee_exception
import io
import requests
import pandas as pd
from typing import List
from shapely.geometry import Polygon
import multiprocessing
from pathlib import Path
import logging