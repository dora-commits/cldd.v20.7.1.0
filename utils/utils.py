# Standard Library
import os
import time
import shutil
import collections
from typing import Optional, Callable, List, Tuple, Dict, Any

# Numerical and Data Handling
import numpy as np
import pandas as pd

# Data Processing and Transformation
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, SubsetRandomSampler, ConcatDataset, Subset
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import is_image_file
from imblearn.metrics import geometric_mean_score, classification_report_imbalanced
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors

# Deep Learning Libraries
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.autograd import Variable

# Computer Vision
import torchvision
from torchvision import datasets
from torchvision.models import resnet18, resnet34

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

# External Libraries
import gdown
import zipfile
import keras.utils as image

# Progress Bar
from tqdm import tqdm