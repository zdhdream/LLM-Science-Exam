import pandas as pd
from string import Template
from pathlib import Path

import os

import warnings
warnings.simplefilter("ignore")

import torch
from transformers import pipeline, AutoTokenizer

from tqdm.notebook import tqdm

data_path = Path('/kaggle/input/kaggle-llm-science-exam')

from transformers import AutoModelForCausalLM, AutoTokenizer


