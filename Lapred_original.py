import numpy as np
import os
import sys

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from math import gcd
import copy

from data_process import NSDatasets, collate_fn
from utils import gpu, to_long, Optimizer, StepLR

file_path = os.path.abspath(__file__)
root_path = os.path.dirname(file_path)
model_name = os.path.basename(file_path).split(".")[0]

# config
config = dict()
"""Train"""
config["display_freq"] = 1.0
config["save_freq"] = 1.0
config["val_freq"] = 2
config["epoch"] = 0
config["horovod"] = True
config["num_epochs"] = 31
config["lr"] = [1e-3, 1e-4]
config["lr_epochs"] = [30]
config["lr_func"] = StepLR(config["lr"], config["lr_epochs"])

if "save_dir" not in config :
  config["save_dir"] = os.path.join(root_path, "results", model_name)

if not os.path.isabs(config["save_dir"]) :
  config["save_dir"] = os.path.join(root_path, "results", config["save_dir"])

config["batch_size"] = 512
config["val_batch_size"] = 512
config["workers"] = 0
config["val_workers"] = config["workers"]

# Dataset
config["DATAROOT"] = './nuscenes/dataset'
# Preprocess
config["preprocess"] = True # whether use preprocess or not
config["preprocess_train"] = os.path.join(root_path, "preprocess", "train_lapred_orig.p")
config["preprocess_val"] = os.path.join(root_path, "preprocess", "train_val_lapred_orig.p")
config['preprocess_test'] = os.path.join(root_path, "preprocess", 'val_lapred_orig.p')

# Lane
config["num_points"] = 20
config["lane_horizon"] = 40.0
config["lane_radius"] = 100
config['outgo_ratio'] = 2
config['incom_ratio'] = 1
config['select_lane'] = 40
config["lane_radius"] = 5
config['lane'] = 15
config["num_points"] = 50
config['lane_forward_length'] = 80
config['lane_backward_length'] = 20

# Model
config["pred_range"] = [-100.0, 100.0, -100.0, 100.0]
config["train_size"] = 2
config["pred_step"] = 1
config["pred_size"] = 6
config["num_preds"] = config["pred_size"] * 2
config["num_mods"] = 10

config['n_agent_p'] = 128
config['n_lane_p'] = 128
config['n_tfe_b'] = 128
config['n_la_b'] = 128
config['n_mtp_b'] = 128

config['reg_lambda'] = 1.
config['lane_off_lambda'] = 1.0
config['l_cls_lambda'] = 0.3
config['cls_lambda'] = 1.0

cp = lambda x: copy.deepcopy(x)

class Linear(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32, act=True):
        super(Linear, self).__init__()
        assert (norm in ['GN', 'BN', 'SyncBN'])

        self.linear = nn.Linear(n_in, n_out, bias=False)

        if norm == 'GN':
            self.norm = nn.GroupNorm(ng,n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class LinearRes(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32):
        super(LinearRes, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.linear1 = nn.Linear(n_in, n_out, bias=False)
        self.linear2 = nn.Linear(n_out, n_out, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if norm == 'GN':
            self.norm1 = nn.GroupNorm(ng,n_out)
            self.norm2 = nn.GroupNorm(ng,n_out)
        elif norm == 'BN':
            self.norm1 = nn.BatchNorm1d(n_out)
            self.norm2 = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        if n_in != n_out:
            if norm == 'GN':
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.GroupNorm(ng,n_out))
            elif norm == 'BN':
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.transform = None

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)

        if self.transform is not None:
            out += self.transform(x)
        else:
            out += x

        out = self.relu(out)
        return out

class Conv1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Conv1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.conv = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, padding=(int(kernel_size) - 1) // 2, stride=stride, bias=False)

        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out

class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config

        self.tfeblock = TFEblock(config)
        self.lablock = LAblock(config)
        self.agent_process = Agent_Process(config)
        self.lane_agent_process = cp(self.agent_process)
        self.lane_process = Lane_Process(config)

        self.mtpblock = MTPblock(config)

    def forward(self, data) :
        # construct agent feature
        agents, agent_idcs, source_agent = get_agents(gpu(data["feats"]))
        agent_ctrs = gpu(data["ctrs"])

        agent_feat = self.agent_process(agents,agent_idcs)

        # lane and agent feature
        lane_feat, label, nearby_trajs = get_lanes(to_long(gpu(data["map_info"])),self.config)

        gl_feats = self.lane_process(lane_feat)

        n_agent_feat = self.lane_agent_process(nearby_trajs)

        tfe_feat = self.tfeblock( \
            agent_feat, gl_feats, n_agent_feat)

        att_w, att_feat = self.lablock(tfe_feat)

        # prediction
        out = self.mtpblock(agent_feat, att_feat, agent_ctrs, source_agent)
        rot, orig = gpu(data["rot"]), gpu(data["orig"])

        out['lane_cls'] =[]
        out['orig_reg'] = out['reg']
        for i in range(len(out["reg"])):
          out["reg"][i] = torch.matmul(out["reg"][i], rot[i]) + orig[i].view(1, 1, -1)
          out['lane_cls'].append(att_w[i])
        return out

def get_agents(agents) :
    batch_size = len(agents)
    num_agents = [len(x) for x in agents]

    source_agent =[x[0:1] for x in agents]
    source_agent = torch.cat(source_agent, 0)

    agents = [x.transpose(1, 2) for x in agents]
    agents = torch.cat(agents, 0)

    agent_idcs = []
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_agents[i]).to(agents.device)
        agent_idcs.append(idcs)
        count += num_agents[i]
    return agents, agent_idcs, source_agent


def get_lanes(map_info,config):
    batch_size = len(map_info)

    lane_feats = [x['lane_feats'].transpose(1, 2).to(torch.float32) for x in map_info]
    lane_feats = torch.cat(lane_feats, 0)

    label = [x['label'] for x in map_info]

    lane_trajs = torch.zeros((batch_size,config['lane'],5,6))
    for i in range(batch_size) :
      x = map_info[i]['nearby_trajs']
      for j in range(len(x)) :
        if len(x[j]) > 0 :
          if len(x[j][0].shape) < 2 :
            lane_trajs[i,j] = x[j]
          else :
            lane_trajs[i,j] = x[j][0]
        else :
          pass

    return lane_feats, label, lane_trajs.cuda()

class Agent_Process(nn.Module):
  def __init__(self, config):
    super(Agent_Process, self).__init__()
    self.config = config
    norm = "BN"

    n_in = 6
    n_agent_p = config['n_agent_p']
    per_time = []

    self.conv = nn.Sequential( \
        Conv1d(n_in,n_agent_p,2,1,'BN'), \
        Conv1d(n_agent_p,n_agent_p,2,1,'BN') \
        )

    self.lstm = nn.LSTM(n_agent_p,n_agent_p,1,batch_first=True)

  def init_state(self,bs) :
    state = (torch.zeros(1,bs,self.config['n_agent_p']).cuda(), \
        torch.zeros(1,bs,self.config['n_agent_p']).cuda())
    return state

  def forward(self, agents, agent_idcs = None) :
    if agent_idcs is None :
      tgt_agent = agents.transpose(-2,-1).flatten(0,1)
    else :
      tgt_agent = []
      for i in range(len(agent_idcs)) : # 총 시나리오 수
        idcs = agent_idcs[i]
        tgt_agent.append(agents[idcs][0:1])
      tgt_agent = torch.cat(tgt_agent,0)

    out = tgt_agent

    out = self.conv(out)

    state_tuple = self.init_state(out.size(0))

    out = out.transpose(1,2)
    out, h_c = self.lstm(out,state_tuple)

    out = out[:,-1]

    return out

class Lane_Process(nn.Module):
  def __init__(self, config):
    super(Lane_Process, self).__init__()
    self.config = config

    n_in = 2
    n_lane_p = config['n_lane_p']

    self.conv = nn.Sequential( \
        Conv1d(n_in,n_lane_p,3,1,'BN'), \
        Conv1d(n_lane_p,n_lane_p,3,1,'BN'), \
        Conv1d(n_lane_p,n_lane_p,3,1,'BN'), \
        Conv1d(n_lane_p,n_lane_p,3,1,'BN'), \
        )

    self.lstm = nn.LSTM(n_lane_p,n_lane_p,1,batch_first=True)

  def init_state(self,bs) :
    state = (torch.zeros(1,bs,self.config['n_lane_p']).cuda(), \
        torch.zeros(1,bs,self.config['n_lane_p']).cuda())
    return state

  def forward(self, lane_feat):
    # local lane_feat
    out = lane_feat

    out = self.conv(out)

    state_tuple = self.init_state(out.size(0))

    out = out.transpose(1,2)
    out, h_c = self.lstm(out,state_tuple)

    out = out[:,-1]

    return out

class TFEblock(nn.Module):
  def __init__(self, config):
      super(TFEblock, self).__init__()
      self.config = config
      norm = "BN"
      ng = config['n_tfe_b']
      n_tfe_b = config['n_tfe_b']

      self.merge = nn.Sequential(
          nn.Linear(config['n_agent_p']*2+config['n_lane_p'], n_tfe_b), \
          nn.ReLU(inplace=True), \
          Linear(n_tfe_b, n_tfe_b, norm=norm, ng=ng, act=False), \
          nn.Linear(n_tfe_b, n_tfe_b), \
          nn.ReLU(inplace=True), \
          Linear(n_tfe_b, n_tfe_b, norm=norm, ng=ng, act=False)
          )

      self.root_path = os.path.dirname(file_path)

  def forward(self, actor_feat, lane_feat, n_actor_feat):
      actor_feat_lane = actor_feat.repeat([self.config["lane"],1])

      tfe_feat=torch.cat([actor_feat_lane, lane_feat, n_actor_feat], 1)
      tfe_feat=self.merge(tfe_feat)

      return tfe_feat

class LAblock(nn.Module):
  def __init__(self, config):
    super(LAblock, self).__init__()
    self.config = config
    norm = "BN"
    ng = config['n_la_b']
    n_lane_p = config['n_lane_p']
    n_la_b = config['n_la_b']
    n_lane = config["lane"]

    self.att = nn.Sequential( \
        nn.Linear(n_lane_p*n_lane, n_la_b), \
        nn.ReLU(inplace=True), \
        Linear(n_la_b, n_la_b, norm=norm, ng=ng, act=False), \
        nn.Linear(n_la_b, n_lane) \
        )

    self.soft=nn.Softmax(dim=1)

  def forward(self, tfe_feat):
    out=tfe_feat
    out=out.view(-1,self.config["lane"],self.config['n_lane_p'])

    out = out.flatten(1,2)
    out= self.att(out)
    att_w = self.soft(out).view(-1,1)
    att_feat=(att_w*tfe_feat).view(-1, self.config["lane"],self.config['n_la_b'])
    att_w = att_w.view(-1, self.config["lane"] )
    att_feat=att_feat.sum(1)
    # att_feat = self.feat(att_feat)

    return att_w, att_feat

class MTPblock(nn.Module):
    def __init__(self, config):
        super(MTPblock, self).__init__()
        self.config = config
        norm = "BN"
        ng = config['n_mtp_b']
        
        n_mtp_b = config['n_mtp_b']
        n_agent_p = config['n_agent_p']
        n_la_b = config['n_la_b']

        pred = []
        for i in range(config["num_mods"]) :
          pred.append( \
              nn.Sequential(
                nn.Linear(n_agent_p+n_la_b,n_mtp_b), \
                nn.ReLU(inplace=True), \
                LinearRes(n_mtp_b, n_mtp_b, norm=norm, ng=ng), \
                nn.Linear(n_mtp_b, 2*config["num_preds"]), \
            )
          )
        self.pred = nn.ModuleList(pred)

        self.att_dest = AttDest(n_mtp_b,n_agent_p,n_la_b)
        self.cls = nn.Sequential(LinearRes(n_mtp_b, n_mtp_b, norm=norm, ng=ng), \
            nn.Linear(n_mtp_b, 1))

    def forward(self, source_feat, att_feat, actor_ctrs, source_actor):
        integ_feat = torch.cat([source_feat, att_feat],1)
        preds = []
        for i in range(len(self.pred)):
          preds.append(self.pred[i](integ_feat))

        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size(0), reg.size(1), -1, 2)

        cls_preds = []
        for i in range(len(self.pred)):
          cls_preds.append(self.pred[i](integ_feat))

        cls_reg = torch.cat([x.unsqueeze(1) for x in cls_preds], 1)
        cls_reg = cls_reg.view(-1, self.config['num_preds']*2)

        source_actor = source_actor[:,:,:2].repeat((self.config["num_mods"], 1,1)).view(cls_reg.size(0), -1)

        cls_feat = torch.cat((source_actor, cls_reg),1)

        actor_ctrs = [x[0:1] for x in actor_ctrs]

        dest_ctrs = reg[:, :, -1].detach()
        feats = self.att_dest(integ_feat, cls_feat, torch.cat(actor_ctrs, 0), dest_ctrs)
        cls = self.cls(feats).view(-1, self.config["num_mods"])

        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), -1, 2)

        out = dict()
        out["cls"], out["reg"] = [], []
        for i in range(len(reg)):
          out["cls"].append(cls[i])
          out["reg"].append(reg[i])
        return out

class AttDest(nn.Module):
  def __init__(self, n_mtp_b, n_agent_p, n_la_b):
    super(AttDest, self).__init__()
    norm = "BN"
    ng = n_mtp_b

    self.dist = nn.Sequential(
        nn.Linear(2, n_mtp_b),
        nn.ReLU(inplace=True),
        Linear(n_mtp_b, n_mtp_b, norm=norm, ng=ng),
    )

    self.cls = nn.Sequential(
        LinearRes(2 * 17, n_mtp_b, norm=norm, ng=ng),
        Linear(n_mtp_b, n_mtp_b, norm=norm, ng=ng),
    )

    self.agt = Linear(2*n_mtp_b + n_agent_p + n_la_b, n_mtp_b, norm=norm, ng=ng)

  def forward(self, agts, cls_feat,agt_ctrs, dest_ctrs):
    n_agt = agts.size(-1)
    num_mods = dest_ctrs.size(1)

    dist = (agt_ctrs.unsqueeze(1) - dest_ctrs).view(-1, 2)
    dist = self.dist(dist)
    agts = agts.unsqueeze(1).repeat(1, num_mods, 1).view(-1, n_agt)
    agts = agts.view(-1,n_agt)
    cls_feat = self.cls(cls_feat)

    agts = torch.cat((dist, agts, cls_feat), 1)
    agts = self.agt(agts)

    return agts

class PredLoss(nn.Module):
  def __init__(self, config):
    super(PredLoss, self).__init__()
    self.config = config
    self.reg_loss = nn.SmoothL1Loss(reduction="sum")
    self.lane_loss = nn.CrossEntropyLoss()
    self.cls_loss = nn.CrossEntropyLoss()

  def forward(self, out, gt_preds, has_preds, map_info, data):
    cls, reg, lane_cls = \
        out["cls"], out["reg"], out["lane_cls"]
    rot, orig = gpu(data["rot"]), gpu(data["orig"])
    cls = torch.cat([x for x in cls], 0)
    cls = cls.view(-1, self.config["num_mods"])

    reg = torch.cat([x for x in reg], 0)
    reg = reg.view(-1, self.config["num_mods"], reg.size(1), reg.size(2))

    lane_cls = torch.cat([x for x in lane_cls], 0)
    lane_cls = lane_cls.view(cls.size(0),-1)

    lane_off_gt = gt_preds = torch.cat([x[0:1] for x in gt_preds], 0)
    lane_off_has = has_preds = torch.cat([x[0:1] for x in has_preds], 0)

    lane_label = []
    lane_feats = []
    for x in map_info:
      if x['label'] == [90] :
        lane_label.append(90)
      else :
        lane_label.append(x['label'])
      lane_feats.append(x['lane_feats'].to( \
          torch.float32).to(has_preds.device))
    lane_labels = torch.tensor(lane_label).to(has_preds.device)

    loss_out = dict()
    zero = 0.0 * (cls.sum() + reg.sum())
    loss_out["cls_loss"] = zero.clone()
    loss_out["num_cls"] = 0
    loss_out["reg_loss"] = zero.clone()
    loss_out["num_reg"] = 0
    loss_out["lane_cls_loss"] = zero.clone()
    loss_out["num_lane_cls"] = 0
    loss_out["lane_off_loss"] = zero.clone()
    loss_out["num_lane_off"] = 0

    num_mods, num_preds = self.config["num_mods"], self.config["num_preds"]
    # assert(has_preds.all())

    last = has_preds.float() + \
        0.1 * torch.arange(num_preds).float().to(has_preds.device) / \
        float(num_preds)
    max_last, last_idcs = last.max(1)
    mask = max_last > 1.0

    cls = cls[mask]
    reg = reg[mask]
    gt_preds = gt_preds[mask]
    has_preds = has_preds[mask]
    last_idcs = last_idcs[mask]

    row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
    dist = []
    for j in range(num_mods) :
      dist.append(torch.sqrt(((reg[row_idcs,j,last_idcs] - \
          gt_preds[row_idcs, last_idcs])**2).sum(1)))
    dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
    min_dist, min_idcs = dist.min(1)
    row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)

    _, cls_tar = (torch.square(reg - gt_preds.unsqueeze(1)).sum(-1).sum(-1)).min(1)

    loss_out["cls_loss"] += self.cls_loss(cls, cls_tar)
    loss_out["num_cls"] += 1

    reg = reg[row_idcs, min_idcs]
    loss_out["reg_loss"] += self.reg_loss(
        reg[has_preds], gt_preds[has_preds]
    )
    loss_out["num_reg"] += has_preds.sum().item()

    for i in range(len(lane_cls)):
      if lane_labels[i] != 90:
        ### lane class loss
        loss_out["lane_cls_loss"] = \
            loss_out["lane_cls_loss"] + \
            self.lane_loss(lane_cls[i:i + 1],lane_labels[i].view(-1)) \
            * (lane_labels[i].view(-1) + 1)
        loss_out["num_lane_cls"] = loss_out["num_lane_cls"] + 1

        ### lane off loss
        if mask[i] :
          _lane = lane_feats[i][lane_labels[i]]
          _lane = torch.matmul(_lane,rot[i])+orig[i].view(1,-1)
          _reg = reg[i]
          _gt = gt_preds[i]
          norm_l_reg = torch.norm(_lane.unsqueeze(0) - \
              _reg.unsqueeze(1),dim=-1).min(-1)[0]
          norm_l_gt = torch.norm(_lane.unsqueeze(0) - \
              _gt.unsqueeze(1),dim=-1).min(-1)[0]
          check = torch.ge(norm_l_reg,norm_l_gt).type(torch.float)
          diff = norm_l_reg-norm_l_gt
          lane_off_loss = (diff*check)[has_preds[i]].mean()
          loss_out['lane_off_loss'] = loss_out['lane_off_loss'] + lane_off_loss
          loss_out['num_lane_off'] += 1

    return loss_out


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.pred_loss = PredLoss(config)

    def forward(self, out, data) :
        loss_out = self.pred_loss(out, gpu(data["gt_preds"]), \
            gpu(data["has_preds"]), gpu(data["map_info"]), data)
        loss_out["loss"] = \
            self.config['cls_lambda']* \
            loss_out["cls_loss"]/(loss_out["num_cls"]+1e-10) \
            + self.config['l_cls_lambda']* \
            loss_out["lane_cls_loss"]/(loss_out["num_lane_cls"]+1e-10) \
            + self.config['reg_lambda']* \
            loss_out["reg_loss"]/(loss_out["num_reg"]+1e-10) \
            + self.config['lane_off_lambda']* \
            loss_out["lane_off_loss"]/(loss_out["num_lane_off"]+1e-10)
        return loss_out


class PostProcess(nn.Module):
    def __init__(self, config):
        super(PostProcess, self).__init__()
        self.config = config

    def forward(self, out,data):
        post_out = dict()
        reg = torch.cat([x for x in out["reg"]], 0)

        post_out["preds"] = [reg.view(-1, self.config["num_mods"], reg.size(1), reg.size(2)).detach().cpu().numpy()]
        post_out["gt_preds"] = [x[0:1].numpy() for x in data["gt_preds"]]
        post_out["has_preds"] = [x[0:1].numpy() for x in data["has_preds"]]
        return post_out

    def append(self, metrics, loss_out, post_out=None) :
        if len(metrics.keys()) == 0:
            for key in loss_out:
                if key != "loss":
                    metrics[key] = 0.0

            for key in post_out:
                metrics[key] = []

        for key in loss_out:
            if key == "loss":
                continue
            if isinstance(loss_out[key], torch.Tensor):
                metrics[key] += loss_out[key].item()
            else:
                metrics[key] += loss_out[key]

        for key in post_out:
            metrics[key] += post_out[key]

        return metrics

    def display(self, metrics, dt, epoch, case, lr=None, writer=None):
        """Every display-iters print training/val information"""
        if lr is not None:
            print("Epoch %3.3f, lr %.5f, time %3.2f" % (epoch, lr, dt))
        else:
            print(
                "************************* Validation, time %3.2f *************************"
                % dt
            )

        cls = metrics["cls_loss"] / (metrics["num_cls"] + 1e-10)
        reg = metrics["reg_loss"] / (metrics["num_reg"] + 1e-10)
        loss = cls + reg

        preds = np.concatenate(metrics["preds"], 0)
        gt_preds = np.concatenate(metrics["gt_preds"], 0)
        has_preds = np.concatenate(metrics["has_preds"], 0)
        ade1, fde1, ade5, fde5, ade, fde, min_idcs = pred_metrics(preds, gt_preds, has_preds)

        print(
            "%s loss %2.4f %2.4f %2.4f, ade1 %2.4f, fde1 %2.4f, ade5 %2.4f, fde5 %2.4f, ade %2.4f, fde %2.4f"
            % (case,loss, cls, reg, ade1, fde1, ade5, fde5, ade, fde)
        )
        if writer is not None :
          writer.add_scalar('Loss/epoch/'+case,loss,epoch)
          writer.add_scalar('RegLoss/epoch/'+case,reg,epoch)
          writer.add_scalar('ClsLoss/epoch/'+case,cls,epoch)
          writer.add_scalar('ADE_1/epoch/'+case,ade1,epoch)
          writer.add_scalar('FDE_1/epoch/'+case,fde1,epoch)
          writer.add_scalar('ADE_5/epoch/'+case,ade5,epoch)
          writer.add_scalar('FDE_5/epoch/'+case,fde5,epoch)
          writer.add_scalar('ADE_10/epoch/'+case,ade,epoch)
          writer.add_scalar('FDE_10/epoch/'+case,fde,epoch)
        print()


def pred_metrics(preds, gt_preds, has_preds):
    assert has_preds.all()
    preds = np.asarray(preds, np.float32)
    gt_preds = np.asarray(gt_preds, np.float32)

    """batch_size x num_mods x num_preds"""
    err = np.sqrt(((preds - np.expand_dims(gt_preds, 1)) ** 2).sum(3))
    ade1 = err[:, 0].mean()
    fde1 = err[:, 0, -1].mean()

    min_idcs = err[:, :5, -1].argmin(1)
    row_idcs = np.arange(len(min_idcs)).astype(np.int64)

    ade_min_idcs = err[:, :5].mean(2).argmin(1)
    ade_row_idcs = np.arange(len(ade_min_idcs)).astype(np.int64)
    err5 = err[:, :5][row_idcs, ade_min_idcs]
    ade5 = err5.mean()
    err5 = err[:, :5][row_idcs, min_idcs]
    fde5 = err5[:, -1].mean()

    ade_min_idcs = err[:, :].mean(2).argmin(1)
    ade_row_idcs = np.arange(len(ade_min_idcs)).astype(np.int64)
    min_idcs = err[:, :, -1].argmin(1)
    row_idcs = np.arange(len(min_idcs)).astype(np.int64)
    err10 = err[row_idcs, ade_min_idcs]
    ade = err10.mean()
    err = err[row_idcs, min_idcs]
    fde = err[:, -1].mean()

    return ade1, fde1, ade5, fde5, ade, fde, min_idcs


def get_model(args=None,preproc=None):
  if args.trial is not None :
    config['save_dir'] = config['save_dir']+'_{}'.format(args.trial)
  net = Net(config)
  net = net.cuda()

  loss = Loss(config).cuda()
  post_process = PostProcess(config).cuda()

  params = net.parameters()
  opt = Optimizer(params, config)


  return config, NSDatasets, collate_fn, net, loss, post_process, opt
