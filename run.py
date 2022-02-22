import os

os.umask(0)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import numpy as np
import random
import sys
import time
import shutil
from importlib import import_module
from tqdm import tqdm
import horovod.torch as hvd

import torch
from torch.utils.data import Sampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from utils import Logger, to_numpy, load_weight

hvd.init()

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model", default="Lapred_original", type=str)

parser.add_argument("--eval", '-e', default=False, action="store_true")
parser.add_argument("--ckpt", '-c', default=0, type=int)
parser.add_argument("--gpu", '-g', default=0, type=int)
parser.add_argument("--trial", '-t', default=None, type=int)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

def main() :

  seed = hvd.rank()
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  model = import_module(args.model)
  config, Dataset, collate_fn, net, loss, post_process, opt = model.get_model(args)

  if config["horovod"] :
    opt.opt = hvd.DistributedOptimizer(opt.opt, \
        named_parameters=net.named_parameters())

  if args.ckpt or args.eval :
    if args.ckpt == 0 :
      ckpt_path = '{}.000.ckpt'.format(config['num_epochs'])
    else :
      ckpt_path = '{}.000.ckpt'.format(args.ckpt)
    if not os.path.isabs(ckpt_path) :
      ckpt_path = os.path.join(config["save_dir"], ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    load_weight(net, ckpt["state_dict"])

  if args.eval :
    # Data loader for evaluation
    dataset = Dataset(config["DATAROOT"], 'test',config, train=False)
    val_sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = DataLoader(dataset, batch_size=config["val_batch_size"], \
        num_workers=config["val_workers"], sampler=val_sampler, \
        collate_fn=collate_fn, pin_memory=True, drop_last=False)

    hvd.broadcast_parameters(net.state_dict(), root_rank=0)
    val(config, val_loader, net, loss, post_process, 999)
    return

  # Create log and copy all code
  save_dir = config["save_dir"]
  log = os.path.join(save_dir, "log")
  writer = SummaryWriter(os.path.join(save_dir,'tensorboard'))
  if hvd.rank() == 0 :
    if not os.path.exists(save_dir) :
      os.makedirs(save_dir)
    sys.stdout = Logger(log)

    src_dirs = [root_path]
    dst_dirs = [os.path.join(save_dir, "files")]
    for src_dir, dst_dir in zip(src_dirs, dst_dirs) :
      files = [f for f in os.listdir(src_dir) if f.endswith(".py")]
      if not os.path.exists(dst_dir) :
        os.makedirs(dst_dir)
      for f in files :
        shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

  # Data loader for training
  dataset = Dataset(config["DATAROOT"], 'train',config, train=True)
  train_sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())

  train_loader = DataLoader(dataset, batch_size=config["batch_size"], \
      num_workers=config["workers"], sampler=train_sampler, \
      collate_fn=collate_fn, pin_memory=True, \
      worker_init_fn=worker_init_fn, drop_last=True)

  # Data loader for evaluation
  dataset = Dataset(config["DATAROOT"], 'val', config, train=False)

  val_sampler = DistributedSampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
  val_loader = DataLoader(dataset, batch_size=config["val_batch_size"], \
      num_workers=config["val_workers"], sampler=val_sampler, \
      collate_fn=collate_fn, pin_memory=True)

  hvd.broadcast_parameters(net.state_dict(), root_rank=0)
  hvd.broadcast_optimizer_state(opt.opt, root_rank=0)

  epoch = config["epoch"]
  remaining_epochs = int(np.ceil(config["num_epochs"] - epoch))
  for i in range(remaining_epochs) :
    train(epoch + i, config, train_loader, net, loss, post_process, \
        opt, val_loader,writer=writer)


def worker_init_fn(pid) :
    np_seed = hvd.rank() * 1024 + int(pid)
    np.random.seed(np_seed)
    random_seed = np.random.randint(2 ** 32 - 1)
    random.seed(random_seed)


def train(epoch, config, train_loader, net, loss, post_process, \
    opt, val_loader=None,writer=None) :
  train_loader.sampler.set_epoch(int(epoch))
  net.train()

  num_batches = len(train_loader)
  epoch_per_batch = 1.0 / num_batches
  save_iters = int(np.ceil(config["save_freq"] * num_batches))
  display_iters = int(np.ceil(config["display_freq"] * num_batches))
  val_iters = int(np.ceil(config["val_freq"] * num_batches))

  start_time = time.time()
  metrics = dict()
  print('training process')
  with torch.autograd.set_detect_anomaly(True) :
    for i, data in tqdm(enumerate(train_loader), disable=hvd.rank()):
      epoch += epoch_per_batch
      data = dict(data)

      output = net(data)
      loss_out = loss(output, data)
      post_out = post_process(output, data)
      post_process.append(metrics, loss_out, post_out)

      opt.zero_grad()
      loss_out["loss"].backward()
      lr = opt.step(epoch)

      num_iters = int(np.round(epoch * num_batches))
      if hvd.rank() == 0 and (num_iters % save_iters == 0 or epoch >= config["num_epochs"]) :
        save_ckpt(net, opt, config["save_dir"], epoch)

      if num_iters % display_iters == 0:
        dt = time.time() - start_time
        if hvd.rank() == 0 :
          post_process.display(metrics, dt, epoch, 'train', lr, writer=writer)
        start_time = time.time()
        metrics = dict()

      if num_iters % val_iters == 0 :
        val(config, val_loader, net, loss, post_process, epoch,writer=writer)

      if epoch >= config["num_epochs"] :
        val(config, val_loader, net, loss, post_process, epoch,writer=writer)
        return


def val(config, data_loader, net, loss, post_process, epoch, writer=None):
  net.eval()

  start_time = time.time()
  metrics = dict()
  print('validation process')
  stores = []
  for i, data in enumerate(data_loader) :
    data = dict(data)
    with torch.no_grad() :
      output = net(data)
      loss_out = loss(output, data)
      post_out = post_process(output, data)
      post_process.append(metrics, loss_out, post_out)

      if args.eval :
        rot,orig = np.stack(data['rot'],0), np.stack(data['orig'],0)
        lane = [x['lane_feats'].transpose(1, 2).to(torch.float32) \
            for x in data['map_info']]
        lane = np.stack(lane,0)
        lane = np.rollaxis(lane,-1,-2)
        lane = np.matmul(lane,np.expand_dims(rot,1)) + np.expand_dims(np.expand_dims(orig,1),1)
        label = [x['label'] for x in data['map_info']]
        agent = [x[0] for x in data['feats']]
        agent = np.stack(agent, 0)[0,:,:2]
        agent = np.matmul(agent,rot)+np.expand_dims(orig,1)
        idx = 0
        for i in range(len(output['reg'])) :
          store = dict()
          store['reg'] = to_numpy(output['reg'][i].detach().cpu().numpy())
          store['cls'] = to_numpy(output['cls'][i].detach().cpu().numpy())
          store['src'] = agent[i]
          store['feats'] = to_numpy(data['feats'][i])
          store['lane'] = lane[i]
          store['label'] = label[i]
          agents = np.array(data['feats'][i][:,:,:2])
          agents = np.matmul(agents, \
              np.expand_dims(rot[i],0))+ \
              np.expand_dims(np.expand_dims(orig[i],0),0)
          store['agents'] = agents
          store['gt_preds'] = to_numpy(data['gt_preds'][i][0])
          store['has_preds'] = to_numpy(data['has_preds'][i])
          store['ctrs'] = to_numpy(data['ctrs'][i])
          store['rot'] = to_numpy(data['rot'][i])
          store['orig'] = to_numpy(data['orig'][i])
          store['ins_sam'] = to_numpy(data['ins_sam'][i])
          store['idx'] = idx

          stores.append(store)

  if args.eval :
    import pickle as pkl
    with open('stores.pkl','wb') as f :
      pkl.dump(stores,f)

  dt = time.time() - start_time
  if hvd.rank() == 0:
    post_process.display(metrics, dt, epoch, 'val',writer=writer)
  net.train()

def save_ckpt(net, opt, save_dir, epoch) :
  if not os.path.exists(save_dir) :
    os.makedirs(save_dir)

  state_dict = net.state_dict()
  for key in state_dict.keys() :
    state_dict[key] = state_dict[key].cpu()

  save_name = "%3.3f.ckpt" % epoch
  torch.save({"epoch": epoch, "state_dict": state_dict, "opt_state": opt.opt.state_dict()}, \
      os.path.join(save_dir, save_name))

if __name__ == "__main__":
  main()
