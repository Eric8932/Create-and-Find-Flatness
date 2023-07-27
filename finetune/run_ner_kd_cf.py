"""
  This script provides an example for NER when applied KD+CFO.
"""


import sys
import os
import random
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch import autograd
from torch.autograd import Variable
from copy import deepcopy
import numpy as np


uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.embeddings import *
from uer.encoders import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.optimizers import *
from uer.utils.constants import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed
from uer.utils.logging import init_logger
from uer.utils.tokenizers import *
from uer.model_saver import save_model
from uer.opts import finetune_opts,tokenizer_opts
from finetune.pytorchtools import EarlyStopping


class NerTagger(nn.Module):
    def __init__(self, args):
        super(NerTagger, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.output_layer = nn.Linear(args.hidden_size, self.labels_num)
        self.crf_target = args.crf_target
        self.temperature = args.temperature
        self.alpha = args.alpha
        self.beta = args.beta
        self.device = args.device
        if args.crf_target:
            from torchcrf import CRF
            self.crf = CRF(self.labels_num, batch_first=True)
            self.seq_length = args.seq_length
        self.dropout = nn.Dropout(0.1)
        self.lamda = args.lamda

    def forward(self, src, tgt, seg,tgt_mask,old_logits=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size x seq_length]
            seg: [batch_size x seq_length]
        Returns:
            loss: Sequence labeling loss.
            logits: Output logits.
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        output = self.dropout(output)

        # Target.
        logits = self.output_layer(output)
        if old_logits == None:
            if self.crf_target:
                tgt_mask = seg.type(torch.uint8)
                pred = self.crf.decode(logits, mask=tgt_mask)
                for j in range(len(pred)):
                    while len(pred[j]) < self.seq_length:
                        pred[j].append(self.labels_num - 1)
                pred = torch.tensor(pred).contiguous().view(-1)
                if tgt is not None:
                    loss = -self.crf(F.log_softmax(logits, 2), tgt, mask=tgt_mask, reduction='mean')
                    return loss, pred
                else:
                    return None, pred
            else:
                tgt_mask = tgt_mask.contiguous().view(-1).float()
                logits = logits.contiguous().view(-1, self.labels_num)
                pred = logits.argmax(dim=-1)
                if tgt is not None:
                    tgt = tgt.contiguous().view(-1, 1)#batch_size*seq_length*1
                    one_hot = torch.zeros(tgt.size(0), self.labels_num). \
                        to(torch.device(tgt.device)). \
                        scatter_(1, tgt, 1.0)
                    numerator = -torch.sum(nn.LogSoftmax(dim=-1)(logits) * one_hot, 1)
                    numerator = torch.sum(tgt_mask * numerator)
                    denominator = torch.sum(tgt_mask) + 1e-6
                    loss = numerator / denominator
                    return loss, pred
                else:
                    return None, pred
        else:
            tgt_mask = tgt_mask.contiguous().view(-1).float()

            old_logits = old_logits.contiguous().view(-1, self.labels_num)
            logits = logits.contiguous().view(-1, self.labels_num)
            logits = nn.LogSoftmax(dim=-1)(logits)
            kl_loss = nn.KLDivLoss(reduction="batchmean")
            pred = logits.argmax(dim=-1)
            if tgt is not None:
                tgt = tgt.contiguous().view(-1, 1)#batch_size*seq_length*1
                tem_ce = torch.ones_like(tgt)
                tem_ce[tgt==0]=0
                tgt_ce = tgt[tem_ce.bool()].unsqueeze(-1)
                logits_ce = logits[tem_ce.bool().squeeze()]
                tgt_mask = tgt_mask[tgt.contiguous().view(-1) !=0]

                one_hot = torch.zeros(tgt_ce.size(0), self.labels_num). \
                    to(torch.device(tgt_ce.device)). \
                    scatter_(1, tgt_ce, 1.0)
                numerator = -torch.sum(logits_ce * one_hot, 1)
                numerator = torch.sum(tgt_mask * numerator)
                denominator = torch.sum(tgt_mask) + 1e-6
                loss_ce = numerator / denominator
                
                tem_kl = torch.ones_like(tgt)
                tem_kl[tgt!=0]=0
                old_logits = old_logits[tem_kl.bool().squeeze()]
                logits_kl = logits[tem_kl.bool().squeeze()]
                loss_kl = kl_loss(logits_kl,old_logits)
                loss_kl = loss_kl*self.temperature*self.temperature
                
                loss = self.alpha*loss_ce+self.beta*loss_kl
                return loss, pred
            else:
                return None, pred
            
    def compute_logits(self, src, seg):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size x seq_length]
            seg: [batch_size x seq_length]
        Returns:
            loss: Sequence labeling loss.
            logits: Output logits.
        """
        src = src.to(self.device)
        seg = seg.to(self.device)
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        logits = self.output_layer(output)
        return logits

    def estimate_fisher(self, src,tgt,seg, tgt_mask,sample_size,batch_size,device,logger):
        loglikelihood_grads = []
        for (src_batch,tgt_batch,seg_batch,tgt_mask_batch) in batch_loader(batch_size,src,tgt,seg,tgt_mask):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            seg_batch = seg_batch.to(device)
            tgt_mask_batch = tgt_mask_batch.to(device)
            logits_l = []
            logits = F.log_softmax(self.compute_logits(src_batch,seg_batch), dim=-1)
            for i in range(batch_size):
                logit = logits[i]
                tgt_cur = tgt_batch[i]
                tgt_mask_cur = tgt_mask_batch[i]
                t = tgt_mask_cur>0
                logit = logit[t]
                tgt_cur = tgt_cur[t]
                logit = logit[range(len(logit)),tgt_cur]
                logits_l.append(torch.mean(logit.float()))

            for i,l in enumerate(logits_l,1):
                grad_l = []
                gra = autograd.grad(l, self.parameters(),retain_graph= (i<len(logits_l)),allow_unused=True)
                for a in gra:
                    if a is not None:
                        grad_l.append(a.detach().cpu())

                loglikelihood_grads.append(grad_l)

            if len(loglikelihood_grads) >= sample_size // batch_size:
                break
        loglikelihood_grads = zip(*loglikelihood_grads)
 

        loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
        fisher_diagonals = [(g ** 2).mean(0).to(device) for g in loglikelihood_grads]
        param_names = [
            n.replace('.', '__') for n, p in self.named_parameters() 
        ]
        return {n: f for n, f in zip(param_names, fisher_diagonals)}
    
    def consolidate(self, mask_fisher,gamma = None):
        for n, p in self.named_parameters():
            n = n.replace('.', '__')
            self.register_buffer('{}_mean'.format(n), p.data.clone())
            # self.register_buffer('{}_fisher'
            #                     .format(n), fisher[n].data.clone())
            self.register_buffer('{}_fisher'
                                    .format(n), mask_fisher[n].data.clone())

    def ewc_loss(self, device):
        try:
            losses = []
            for n, p in self.named_parameters():
                n = n.replace('.', '__')
                mean = getattr(self, '{}_mean'.format(n))
                fisher = getattr(self, '{}_fisher'.format(n))
                mean = Variable(mean,requires_grad=True)
                fisher = Variable(fisher,requires_grad = True)
                if "output_layer" in n:
                    p1 = p[:-2].clone()
                    losses.append((fisher * (p1-mean)**2).sum())
                else:
                    losses.append((fisher * (p-mean)**2).sum())
            return (self.lamda/2)*sum(losses).to(device)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1),requires_grad=True).to(device)
            )

def evaluate(args, dataset):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])
    tgt_mask = torch.LongTensor([sample[3] for sample in dataset])

    instances_num = src.size(0)
    batch_size = args.batch_size

    correct,gold_entities_num,pred_entities_num ={},{},{}

    args.model.eval()

    for i, (src_batch, tgt_batch, seg_batch, tgt_mask_batch) in enumerate(batch_loader(batch_size, src, tgt, seg,tgt_mask)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        tgt_mask_batch = tgt_mask_batch.to(args.device)
        loss, pred = args.model(src_batch, tgt_batch, seg_batch,tgt_mask_batch)

        tgt_mask_batch = tgt_mask_batch.contiguous().view(-1, 1)
        gold = tgt_batch.contiguous().view(-1, 1)
        pred = pred.unsqueeze(-1)

        gold = gold[tgt_mask_batch!=0]
        pred = pred[tgt_mask_batch!=0]

        for j in range(gold.size()[0]):
            if gold[j].item() in args.begin_ids:
                if gold[j].item() not in gold_entities_num:
                    gold_entities_num[gold[j].item()]=1
                else:
                    gold_entities_num[gold[j].item()]+=1
        for j in range(pred.size()[0]):
            if pred[j].item() in args.begin_ids and gold[j].item() != args.l2i["[PAD]"]:
                if pred[j].item() not in pred_entities_num:
                    pred_entities_num[pred[j].item()]=1
                else:
                    pred_entities_num[pred[j].item()]+=1

        pred_entities_pos, gold_entities_pos = set(), set()

        for j in range(gold.size()[0]):
            if gold[j].item() in args.begin_ids:
                start = j
                for k in range(j + 1, gold.size()[0]):
                    if gold[k].item() == args.l2i["[PAD]"] or gold[k].item() == args.l2i["O"] or gold[k].item() in args.begin_ids:
                        end = k - 1
                        break
                else:
                    end = gold.size()[0] - 1
                gold_entities_pos.add((start, end,gold[j].item()))

        for j in range(pred.size()[0]):
            if pred[j].item() in args.begin_ids and gold[j].item() != args.l2i["[PAD]"]:
                start = j
                for k in range(j + 1, pred.size()[0]):
                    if pred[k].item() == args.l2i["[PAD]"] or pred[k].item() == args.l2i["O"] or pred[k].item() in args.begin_ids:
                        end = k - 1
                        break
                else:
                    end = pred.size()[0] - 1
                pred_entities_pos.add((start, end,pred[j].item()))

        for entity in pred_entities_pos:
            if entity not in gold_entities_pos:
                continue
            for j in range(entity[0], entity[1] + 1):
                if gold[j].item() != pred[j].item():
                    break
            else:
                if entity[2] not in correct:
                    correct[entity[2]] = 1
                else:
                    correct[entity[2]] += 1

    args.logger.info("Report precision, recall, and f1:")
    eps = 1e-9
    f1_avg = 0.0
    f1_dic = {}
    for a in correct:
        p = correct[a] / (pred_entities_num[a] + eps)
        r = correct[a] / (gold_entities_num[a] + eps)
        f1 = 2 * p * r / (p + r + eps)
        f1_dic[a] = f1
        f1_avg+=f1
        args.logger.info("{}: {:.3f}, {:.3f}, {:.3f}".format(a,p, r, f1))
    if len(correct) == 0:
        f1_avg = 0.0
    else:
        f1_avg/= len(correct)
    args.logger.info("F1_AVG {} for len:{}".format(f1_avg,len(correct)))
    return f1_avg,f1_dic

def train(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch,tgt_mask_batch,task_id,old_logits=None):
    model.train()
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    tgt_mask_batch = tgt_mask_batch.to(args.device)

    if old_logits is not None:
        old_logits = old_logits.to(args.device)

    enable_running_stats(model)
    loss, _ = model(src_batch, tgt_batch, seg_batch,tgt_mask_batch,old_logits)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    disable_running_stats(model)
    loss, _ = model(src_batch, tgt_batch, seg_batch,tgt_mask_batch,old_logits)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)
    loss.backward()

    optimizer.second_step(zero_grad=False)

    if task_id !=0 :
        enable_running_stats(model)
        ewc_loss = model.ewc_loss(args.device)
        if torch.cuda.device_count() > 1:
            ewc_loss = torch.mean(ewc_loss)
        ewc_loss.requires_grad_(True)
        ewc_loss.backward()
    
    optimizer.third_step(zero_grad=True)
    
    if task_id !=0:
        clamp_params(model,task_id)

    scheduler.step()
    return loss

def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location="cpu"), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)

def batch_loader(batch_size, src, tgt, seg,tgt_mask):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size, :]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        tgt_mask_batch = tgt_mask[i * batch_size : (i + 1) * batch_size, :]
        yield src_batch, tgt_batch, seg_batch, tgt_mask_batch
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :, :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        tgt_mask_batch = tgt_mask[instances_num // batch_size * batch_size :, :]
        yield src_batch, tgt_batch, seg_batch, tgt_mask_batch

def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.rstrip("\r\n").split("\t")):
                    columns[column_name] = i
                continue
            line = line.rstrip("\r\n").split("\t")
            labels = line[columns["label"]]
            whether_con = 1
            if "test" in path:
                tgt = [args.l2i[l] if l in args.l2i else args.l2i['O'] for l in labels.split(" ")]
            else:
                tgt = [args.l2i[l] if args.cur_type in l else args.l2i['O'] for l in labels.split(" ")]
                
            text_a = line[columns["text_a"]]
            
            text =text_a.split(" ")
            src_token = []
            label_list = []
            tgt_mask = []
            src_token += [CLS_TOKEN]
            label_list.append(args.l2i[CLS_TOKEN])
            tgt_mask.append(1)
            for i, word in enumerate(text):
                token = args.tokenizer.tokenize(word)
                label_cur = tgt[i]
                src_token += token
                for m in range(len(token)):
                    if m == 0:
                        label_list.append(label_cur)
                        tgt_mask.append(1)
                    else:
                        label_list.append(args.l2i['[PAD]'])
                        tgt_mask.append(0)
            src_token += [SEP_TOKEN]
            label_list.append(args.l2i[SEP_TOKEN])
            tgt_mask.append(1)

            src = args.tokenizer.convert_tokens_to_ids(src_token)
            seg = [1] * len(src)
            tgt = label_list

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                tgt = tgt[: args.seq_length]
                seg = seg[: args.seq_length]
                tgt_mask = tgt_mask[: args.seq_length]
            PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
            while len(src) < args.seq_length:
                src.append(PAD_ID)
                tgt.append(args.l2i['[PAD]'])
                seg.append(0)
                tgt_mask.append(0)
            dataset.append([src, tgt, seg,tgt_mask])

    args.logger.info("dataset length: {}".format(len(dataset)))
    return dataset

def update_register(model,rho):
    model.register_buffer('pre_rho',rho)


def clamp_params(model,task_id):
    with torch.no_grad():
        for i,(n, p) in enumerate(model.named_parameters()):
            n_old = n.replace('.', '__')
            p_origin = getattr(model, '{}_mean'.format(n_old))
            bound_value = getattr(model,'pre_rho')
            p_bound = bound_value*(torch.abs(p_origin))
            p_upper_bound = p_origin + p_bound 
            p_lower_bound = p_origin - p_bound 
            if p.data.shape != p_origin.shape:
                if len(p_origin.shape)==2:
                    tem_upper = (torch.ones([2,p_origin.shape[1]])*1e10).to(p.device)
                    tem_lower = (torch.ones([2,p_origin.shape[1]])*(-1e10)).to(p.device)
                else:
                    tem_upper = (torch.ones([2])*1e10).to(p.device)
                    tem_lower = (torch.ones([2])*(-1e10)).to(p.device)
                p_upper_bound = torch.cat((p_upper_bound,tem_upper),0)
                p_lower_bound = torch.cat((p_lower_bound,tem_lower),0)
            p_upper_mask = p.data > p_upper_bound
            p_lower_mask = p.data < p_lower_bound
            p.data[p_upper_mask] = p_upper_bound[p_upper_mask]
            p.data[p_lower_mask] = p_lower_bound[p_lower_mask]


def compute_dev_loss(args,task_id):
    loss_list = []
    args.model.eval()
    instances = read_dataset(args, args.dev_path.replace(".tsv","_"+str(task_id)+".tsv"))
    src = torch.LongTensor([ins[0] for ins in instances])
    tgt = torch.LongTensor([ins[1] for ins in instances])
    seg = torch.LongTensor([ins[2] for ins in instances])
    tgt_mask = torch.LongTensor([ins[3] for ins in instances])

    for i, (src_batch, tgt_batch, seg_batch,tgt_mask_batch) in enumerate(batch_loader(args.batch_size, src, tgt, seg,tgt_mask)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        tgt_mask_batch = tgt_mask_batch.to(args.device)
        with torch.no_grad():
            loss,_ = args.model(src_batch,tgt_batch,seg_batch,tgt_mask_batch)
            loss_list.append(loss.cpu().item())
    return np.mean(loss_list)

def build_optimizer(args, model):
    if args.optimizer == 'sgd':
        optimizer = str2optimizer[args.optimizer](model.param_groups, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01,"adaptive":args.adaptive,"rho":args.rho},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,"adaptive":args.adaptive,"rho":args.rho},
        ]

        if args.optimizer in ["adamw"]:
            optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
        else:
            optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                      scale_parameter=False, relative_step=False)
    

    return optimizer
    
def build_sechduler(args,optimizer,task_id):
    if args.scheduler in ["steplr"]:
        scheduler = StepLR(optimizer, args.learning_rate, args.epochs[task_id])
    elif args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return scheduler

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)
    tokenizer_opts(parser)


    parser.add_argument("--crf_target", action="store_true",
                        help="Use CRF loss as the target function or not, default False.")

    parser.add_argument("--epochs", nargs='+', type=int,
                    default=[10, 10, 10, 10, 10],
                    help='Epoch number for each task')

    parser.add_argument("--adaptive",action='store_true', help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--rho", default=0.65, type=float, help="Rho parameter for SAM.")

    parser.add_argument("--consolidate",action='store_false',
                help = 'Whether computing the fisher info')
    parser.add_argument("--fisher_estimation_sample_size", type=int, default=1024,
                        help="...")
    parser.add_argument("--lamda", type=int, default=1000,
                        help="coefficient for ewc_loss")
    parser.add_argument("--gamma", type=float, default=0.0,
                        help='F(i+1)=F_now+gamma*F_old')

    parser.add_argument('--tasks', nargs='+', type=str,
                    default=['LOC','ORG','MISC','PER'],
                    help='Task Sequence')

    parser.add_argument('--tasks_order', nargs='+', type=str,
                    default=['0','1','2','3'],
                    help='Task Sequence')
    parser.add_argument('--evaluate_steps', type=int, default=500,
                        help='specific step to evaluate whether current model is the best')

    parser.add_argument('--alpha', type=float, default=1,
                        help='coefficient for CE Loss')
    parser.add_argument('--beta', type=float, default=1,
                        help='coefficient for KD Loss')
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='temperature for SoftMax logits')
    parser.add_argument('--patience', type=int, default=3,
                        help='patience for early stopping')          

    args = parser.parse_args()

    # Load the hyperparameters of the config file.
    args = load_hyperparam(args)

    # Get logger.
    args.logger = init_logger(args)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)
    

    # args.tokenizer = SpaceTokenizer(args)
    args.tokenizer = BertTokenizer(args)
    
    args.tasks_list = args.tasks
    args.tasks = []
    for i in args.tasks_order:
        args.tasks.append(args.tasks_list[int(i)])

    args.begin_ids = []
    args.l2i = {}
    args.l2i['O']=0
    len_l2i = 1

    task_cur = args.tasks[0]
    args.l2i['B-'+task_cur]=len_l2i
    args.begin_ids.append(len_l2i)
    len_l2i+=1
    args.l2i['I-'+task_cur]=len_l2i
    len_l2i+=1
    args.l2i["[PAD]"] = len_l2i
    len_l2i+=1
    args.l2i["[CLS]"] = len_l2i
    len_l2i+=1
    args.l2i["[SEP]"] = len_l2i
    len_l2i+=1

    args.labels_num = len(args.l2i)
    args.cur_type = args.tasks[0]

    instances = read_dataset(args, args.train_path.replace(".tsv","_"+"0"+".tsv"))
    instances_num = len(instances)
    batch_size = args.batch_size
    args.train_steps = int(instances_num * args.epochs[0] / batch_size) + 1

    args.logger.info("Batch size: {}".format(batch_size))
    args.logger.info("The number of training instances: {}".format(instances_num))

    # Build sequence labeling model.
    model = NerTagger(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)
    model = model.to(args.device)

    if torch.cuda.device_count() > 1:
        args.logger.info("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)


    args.model = model
    base_optimizer= build_optimizer(args, model)#SAM
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate)
    scheduler = build_sechduler(args,optimizer,0)
    
    pre_rho = torch.tensor(1.0e100)
    args.logger.info("rho: {}".format(args.rho))
    if args.adaptive:
        args.logger.info("adaptive on")

    args.logger.info("Start training.")
    f1_dic = {}

    args.logger.info("Task: {}".format(" ".join(args.tasks)))
    
    for task_id in range(len(args.tasks)):
        args.cur_type = args.tasks[task_id]
        total_loss, f1, best_f1 = 0.0, 0.0, 0.0
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        if task_id ==0:
            for epoch in range(1, args.epochs[task_id] + 1):
                random.shuffle(instances)
                src = torch.LongTensor([ins[0] for ins in instances])
                tgt = torch.LongTensor([ins[1] for ins in instances])
                seg = torch.LongTensor([ins[2] for ins in instances])
                tgt_mask = torch.LongTensor([ins[3] for ins in instances])

                for i, (src_batch, tgt_batch, seg_batch,tgt_mask_batch) in enumerate(batch_loader(batch_size, src, tgt, seg,tgt_mask)):
                    loss = train(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch,tgt_mask_batch,task_id)
                    total_loss += loss.item()
                    if (i + 1) % args.report_steps == 0:
                        args.logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                        total_loss = 0.0
                    
                dev_loss = compute_dev_loss(args,task_id)
                early_stopping(dev_loss, model)
                if early_stopping.early_stop:
                    args.logger.info("Early Stop At Epoch {}".format(epoch))
                    break
                
            f1,f1_cur_dic = evaluate(args, read_dataset(args, args.test_path))
            f1_dic[task_id] = f1_cur_dic
        else:
            pre_rho = torch.tensor(args.rho)
            update_register(model,pre_rho)

            old_model = deepcopy(model).to(args.device)
            old_model.eval()

            ori_weight = model.output_layer.weight
            ori_bias = model.output_layer.bias
            new_weight=torch.normal(0, 0.02, size=(2,args.hidden_size), requires_grad=True).to(args.device)
            new_bias=torch.normal(0, 0.02, size=(2,1), requires_grad=True).flatten().to(args.device)
            weight = torch.cat((ori_weight,new_weight))
            bias = torch.cat((ori_bias,new_bias))
            model.output_layer.weight = Parameter(weight)  
            model.output_layer.bias = Parameter(bias) 

            print(model.output_layer.weight.shape)

            task_cur = args.tasks[task_id]
            args.l2i['B-'+task_cur]=len_l2i
            args.begin_ids.append(len_l2i)
            len_l2i+=1
            args.l2i['I-'+task_cur]=len_l2i
            len_l2i+=1

            args.labels_num = len(args.l2i)
            model.labels_num = args.labels_num

            instances = read_dataset(args, args.train_path.replace(".tsv","_"+str(task_id)+".tsv"))
            instances_num = len(instances)
            batch_size = args.batch_size
            args.train_steps = int(instances_num * args.epochs[task_id] / batch_size) + 1

            args.logger.info("Batch size: {}".format(batch_size))
            args.logger.info("The number of training instances: {}".format(instances_num))

            base_optimizer= build_optimizer(args, model)#SAM
            optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate)
            scheduler = build_sechduler(args,optimizer,task_id)

            for epoch in range(1, args.epochs[task_id] + 1):
                random.shuffle(instances)
                src = torch.LongTensor([ins[0] for ins in instances])
                tgt = torch.LongTensor([ins[1] for ins in instances])
                seg = torch.LongTensor([ins[2] for ins in instances])
                tgt_mask = torch.LongTensor([ins[3] for ins in instances])
                for i, (src_batch, tgt_batch, seg_batch,tgt_mask_batch) in enumerate(batch_loader(batch_size, src, tgt, seg,tgt_mask)):
                    with torch.no_grad():
                        old_logits = old_model.compute_logits(src_batch,seg_batch)

                        ze = (torch.ones([old_logits.shape[0],old_logits.shape[1],2])*(-1e9)).to(args.device)
                        old_logits = torch.cat((old_logits,ze),-1)
                        old_logits = nn.Softmax(dim=-1)(old_logits/args.temperature)
                        
                    loss = train(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch,tgt_mask_batch,task_id,old_logits)
                    total_loss += loss.item()
                    if (i + 1) % args.report_steps == 0:
                        args.logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                        total_loss = 0.0
  
                dev_loss = compute_dev_loss(args,task_id)
                early_stopping(dev_loss, model)
                if early_stopping.early_stop:
                    args.logger.info("Early Stop At Epoch {}".format(epoch))
                    break
            
            f1,f1_cur_dic = evaluate(args, read_dataset(args, args.test_path))
            f1_dic[task_id] = f1_cur_dic
        if args.consolidate and task_id < len(args.tasks)-1:
            # estimate the fisher information of the parameters and consolidate
            # them in the network.
            args.logger.info(
                '=> Estimating diagonals of the fisher information matrix...',
            )
            random.shuffle(instances)
            src = torch.LongTensor([ins[0] for ins in instances])
            tgt = torch.LongTensor([ins[1] for ins in instances])
            seg = torch.LongTensor([ins[2] for ins in instances])
            tgt_mask = torch.LongTensor([ins[3] for ins in instances])
            model.consolidate(model.estimate_fisher(
                src,tgt,seg, tgt_mask,args.fisher_estimation_sample_size,args.batch_size,args.device,args.logger
            ),args.gamma)
            args.logger.info(' Done!')

    args.logger.info("Summary of F1")
    for a in f1_dic:
        f1_cur_dic = f1_dic[a]
        f1_list = []
        f1_str_list = []
        task_list = []
        b = list(f1_cur_dic.keys())
        b.sort()
        for c in b:
            f1_list.append(f1_cur_dic[c])
            f1_str_list.append(str(f1_cur_dic[c]))
        for i in range(0,a+1):
            task_list.append(args.tasks[i])
        
            
        args.logger.info("F1 on Task {}:{} and Mean:{} ".format(" ".join(task_list)," ".join(f1_str_list),np.mean(f1_list)))

 


if __name__ == "__main__":
    main()
