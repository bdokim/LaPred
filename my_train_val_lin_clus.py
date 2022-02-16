import argparse
import os
import pickle
from importlib import import_module
from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader

from data_process import NSDatasets as Dataset, collate_fn
from utils import to_numpy

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import sys

os.umask(0)

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", default="config", type=str)
parser.add_argument("--trial", '-t', default=None, type=int)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']='2'

model = import_module('Lapred_original')
config, *_ = model.get_model(args)
config["preprocess"] = False

os.makedirs(os.path.dirname(config['preprocess_train']),exist_ok=True)

dataset=Dataset(config["DATAROOT"], 'train_val',config, train=True)
train_loader = DataLoader(dataset, batch_size=1, num_workers=16, shuffle=False, \
    collate_fn=collate_fn, pin_memory=True, drop_last=False)

def to_int16(data):
  if isinstance(data, dict):
    for key in data.keys():
      data[key] = to_int16(data[key])
  if isinstance(data, list) or isinstance(data, tuple):
    data = [to_int16(x) for x in data]
  if isinstance(data, np.ndarray) and data.dtype == np.int64:
    data = data.astype(np.int16)
  return data

stores = [None for x in range(8560)]

for i, data in enumerate(tqdm(train_loader)):
  data = dict(data)
  for j in range(len(data["idx"])) :
    store = dict()
    for key in ["idx","feats","ctrs","orig","theta","rot", \
        "gt_preds","has_preds","ins_sam","map_info"] :
      store[key] = to_numpy(data[key][j])
      if key in ["map_info"]:
        store[key] = to_int16(store[key])
    stores[store["idx"]] = store

f = open(os.path.join(root_path, 'preprocess/train_val_lapred_orig.p'), 'wb')
print(f)
pickle.dump(stores, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()
