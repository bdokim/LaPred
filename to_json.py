import json
import pickle as pkl
import pdb

import matplotlib.pyplot as plt
from nuscenes.eval.prediction.data_classes import Prediction

import numpy as np

def softmax(x) :
  return np.exp(x)/np.sum(np.exp(x))

def to_meta(out) :
  ins,sam = out['ins_sam']
  regs = out['reg']
  prob = softmax(out['cls'])
  return Prediction(ins,sam,regs,prob)

with open('stores.pkl','rb') as f :
  out = pkl.load(f)

preds = []
for i in range(len(out)) :
  preds.append(to_meta(out[i]).serialize())

json.dump(preds,open('./meta.json','w'))

exit()

preds = np.array([x['reg'] for x in out])
gt_preds = np.array([x['gt_preds'] for x in out])
err = np.sqrt(((preds - np.expand_dims(gt_preds, 1)) ** 2).sum(3))
min_idcs = err[:, :5, -1].argmin(1)
row_idcs = np.arange(len(min_idcs)).astype(np.int64)
ade_min_idcs = err[:, :5].mean(2).argmin(1)
ade_row_idcs = np.arange(len(ade_min_idcs)).astype(np.int64)
err5 = err[:,:5][row_idcs,min_idcs]

order = np.argsort(err5.mean(-1))

for i in range(len(order)) :
  idx=order[i]
  lane = out[idx]['lane']
  reg = out[idx]['reg']
  src = out[idx]['src']
  gt_preds = out[idx]['gt_preds']
  actor = out[idx]['actors']
  # label = out[idx]['label']
  for j in range(lane.shape[0]) :
    plt.plot(lane[j,:,0],lane[j,:,1],'y')
  # plt.plot(lane[label,:,0],lane[label,:,1],'g')
  # pdb.set_trace()
  # for j in range(1,len(actor)) :
  #   plt.plot(actor[j,:,0],actor[j,:,1],'o-k')
  for j in range(5) :
    plt.plot(reg[j,:,0],reg[j,:,1],'g')
    plt.plot(reg[j,-1,0],reg[j,-1,1],'go')
  plt.plot(src[:,0],src[:,1],'r')
  plt.plot(gt_preds[:,0],gt_preds[:,1],'b')
  plt.show()
