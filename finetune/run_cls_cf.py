"""
This script provides an example for classification when applied C&F.
"""

from itertools import product
import sys
import os
import random
import argparse
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from functools import reduce
import numpy as np
from copy import deepcopy
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
from sklearn.cluster import KMeans, MiniBatchKMeans


uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.embeddings import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.utils.logging import init_logger
from uer.utils.misc import pooling
from uer.model_saver import save_model
from uer.opts import finetune_opts, tokenizer_opts, adv_opts

dataset_map_list = []

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        # self.labels_num = args.labels_num
        self.pooling_type = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layers_1 = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in args.labels_num_list])
        self.output_layers_2 = nn.ModuleList([nn.Linear(args.hidden_size, labels_num) for labels_num in args.labels_num_list])
        self.dataset_id = 0
        self.lamda = args.lamda

    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        # Target.
        output = pooling(output, seg, self.pooling_type)
        output = torch.tanh(self.output_layers_1[self.dataset_id](output))
        logits = self.output_layers_2[self.dataset_id](output)

        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            else:
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits
        
    def change_dataset(self, dataset_id):
        self.dataset_id = dataset_id

    
    def compute_features(self, src,seg):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = pooling(output, seg, self.pooling_type)
        return output

    def compute_logits(self, src,seg):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        # Target.
        output = pooling(output, seg, self.pooling_type)

        output = torch.tanh(self.output_layers_1[self.dataset_id](output))
        logits = self.output_layers_2[self.dataset_id](output)
        return logits

    def estimate_fisher(self, src,tgt,seg, sample_size, batch_size,device,logger):
        # loglikelihoods = []
        loglikelihood_grads = []
        for (src_batch,tgt_batch,seg_batch,_) in batch_loader(batch_size,src,tgt,seg):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            seg_batch = seg_batch.to(device)


            logits_l = F.log_softmax(self.compute_logits(src_batch,seg_batch), dim=1)[range(batch_size), tgt_batch]

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
            n.replace('.', '__') for n, p in self.named_parameters() if "output_layers" not in n or ("."+str(self.dataset_id)+".") in n
        ]
        return {n: f for n, f in zip(param_names, fisher_diagonals)}
    
    def consolidate(self, mask_fisher,gamma):
        for n, p in self.named_parameters():
            if "output_layers" not in n or ("."+str(self.dataset_id)+".") in n:
                n = n.replace('.', '__')
                self.register_buffer('{}_mean'.format(n), p.data.clone())
                if hasattr(self,'{}_fisher'.format(n)):
                    fisher_old = getattr(self, '{}_fisher'.format(n))
                    fisher_new = fisher_old*gamma + mask_fisher[n].data.clone()
                    self.register_buffer('{}_fisher'
                                    .format(n), fisher_new)
                else:
                    self.register_buffer('{}_fisher'
                                    .format(n), mask_fisher[n].data.clone())
                    
    def ewc_loss(self, device,task_id,replay):
        try:
            losses = []
            for n, p in self.named_parameters():
                if "output_layers" not in n or (self.dataset_id<task_id and self.dataset_id!=dataset_map_list[task_id] and ("."+str(self.dataset_id)+".") in n):
                # retrieve the consolidated mean and fisher information.
                    n = n.replace('.', '__')
                    mean = getattr(self, '{}_mean'.format(n))
                    fisher = getattr(self, '{}_fisher'.format(n))
                    # wrap mean and fisher in variables.
                    mean = Variable(mean,requires_grad=True)
                    fisher = Variable(fisher,requires_grad = True)

                    losses.append((fisher * (p-mean)**2).sum())
            return (self.lamda/2)*sum(losses).to(device)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1),requires_grad=True).to(device)
            )



class Memory(object):
    def __init__(self,args):
        self.examples = []
        self.masks = []
        self.labels = []
        self.tasks = []


    def append(self, example, mask, label, task):
        self.examples.append(example)
        self.masks.append(mask)
        self.labels.append(label)
        self.tasks.append(task)

    def get_random_batch(self, batch_size, task_id=None):
        if task_id is None:
            permutations = np.random.permutation(len(self.labels))
            index = permutations[:batch_size]
            mini_examples = [self.examples[i] for i in index]
            mini_masks = [self.masks[i] for i in index]
            mini_labels = [self.labels[i] for i in index]
        else:
            index = [i for i in range(len(self.labels)) if self.tasks[i] == task_id]
            np.random.shuffle(index)
            index = index[:batch_size]
            mini_examples = [self.examples[i] for i in index]
            mini_masks = [self.masks[i] for i in index]
            mini_labels = [self.labels[i] for i in index]
        return torch.LongTensor(mini_examples), torch.LongTensor(mini_masks), torch.LongTensor(mini_labels)
    def __len__(self):
        return len(self.labels)

def read_dataset(args, path,task_id):
    dataset, columns = [], {}
    tgt_offset = args.offset[task_id]

    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.rstrip("\r\n").split("\t")):
                    columns[column_name] = i
                continue
            line = line.rstrip("\r\n").split("\t")
            tgt = int(line[columns["label"]])+tgt_offset

            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
            if "text_b" not in columns:  # Sentence classification.
                text_a = line[columns["text_a"]]
                src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                seg = [1] * len(src)
            else:  # Sentence-pair classification.
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
            while len(src) < args.seq_length:
                src.append(PAD_ID)
                seg.append(0)
            if args.soft_targets and "logits" in columns.keys():
                dataset.append((src, tgt, seg, soft_tgt))
            else:
                dataset.append((src, tgt, seg))

    return dataset

def sample_dataset(args, dataset,i):
    np.random.seed(args.seed)
    train_idxs = []
    val_idxs = []
    label_num = args.labels_num_list[i]
    offset = args.offset[i]

    for cls in range(offset,offset+label_num):
        idxs= []
        for j in range(len(dataset)):
            if dataset[j][1] == cls:
                idxs.append(j)
        idxs = np.array(idxs)
        np.random.shuffle(idxs)

        train_pool = idxs[:-args.n_val]
        if args.n_labeled < 0:
            train_idxs.extend(train_pool)
        else:
            train_idxs.extend(train_pool[:args.n_labeled])
        val_idxs.extend(idxs[-args.n_val:])

    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    dataset = np.array(dataset)
    train_dataset = list(dataset[train_idxs])
    val_dataset = list(dataset[val_idxs])

    return train_dataset,val_dataset

def remove_dup(l_list):
    l = []
    for a in l_list:
        if a not in l:
            l.append(a)
    return l

def evaluate(args, dataset):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size

    correct = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    args.model.eval()

    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            _, logits = args.model(src_batch, tgt_batch, seg_batch)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()

    args.logger.info("Confusion matrix:")
    args.logger.info(confusion)
    args.logger.info("Report precision, recall, and f1:")

    eps = 1e-9
    for i in range(confusion.size()[0]):
        p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
        r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
        f1 = 2 * p * r / (p + r + eps)
        args.logger.info("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))

    args.logger.info("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
    return correct / len(dataset), confusion


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size : (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size :, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None

def count_labels_num(path):
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.rstrip("\r\n").split("\t")):
                    columns[column_name] = i
                continue
            line = line.rstrip("\r\n").split("\t")
            label = int(line[columns["label"]])
            labels_set.add(label)
    return len(labels_set)


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location="cpu"), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)

def select_samples_to_store(args,model, buffer, trainset, task_id):
    ### ----------- add examples to memory ------------------ ##
    x_list = []
    mask_list = []
    y_list = []
    fea_list = []
    model.eval()
    src = torch.LongTensor([example[0] for example in trainset])
    tgt = torch.LongTensor([example[1] for example in trainset])
    seg = torch.LongTensor([example[2] for example in trainset])
    with torch.no_grad():
        
        for (x,y,mask,_) in batch_loader(args.batch_size, src, tgt, seg, None):
            x = x.to(args.device)
            mask = mask.to(args.device)
            y = y.to(args.device)
            if args.kmeans:
                bert_emb = model.compute_features(x, mask)
                fea_list.append(bert_emb.to("cpu"))
            x_list.append(x.to("cpu"))
            mask_list.append(mask.to("cpu"))
            y_list.append(y.to("cpu"))
            # Kmeans on bert embedding
    x_list = torch.cat(x_list, dim=0).data.cpu().numpy()
    mask_list = torch.cat(mask_list, dim=0).data.cpu().numpy()
    y_list = torch.cat(y_list, dim=0).data.cpu().numpy()
    if args.kmeans:
        fea_list = torch.cat(fea_list, dim=0).data.cpu().numpy()

    # if use KMeans
    if args.kmeans:
        n_clu = int(args.store_ratio * len(x_list))
        estimator = KMeans(n_clusters=n_clu, random_state=args.seed)
        estimator.fit(fea_list)
        label_pred = estimator.labels_
        centroids = estimator.cluster_centers_
        for clu_id in range(n_clu):
            index = [i for i in range(len(label_pred)) if label_pred[i] == clu_id]
            closest = float("inf")
            closest_x = None
            closest_mask = None
            closest_y = None
            for j in index:
                dis = np.sqrt(np.sum(np.square(centroids[clu_id] - fea_list[j])))
                if dis < closest:
                    closest_x = x_list[j]
                    closest_mask = mask_list[j]
                    closest_y = y_list[j]
                    closest = dis

            if closest_x is not None:
                buffer.append(closest_x, closest_mask, closest_y, task_id)
    else:
        permutations = np.random.permutation(len(x_list))
        index = permutations[:int(args.store_ratio * len(x_list))]
        for j in index:
            buffer.append(x_list[j], mask_list[j], y_list[j], task_id)
    args.logger.info("Buffer size:{}".format(len(buffer)))
    args.logger.info(buffer.labels)
    b_lbl = np.unique(buffer.labels)
    for i in b_lbl:
        args.logger.info("Label {} in Buffer: {}".format(i, buffer.labels.count(i)))

def construct_maplist(tasks):#5tasks
    mark = -1
    i=0
    for a in tasks:
        a = a.replace('_exp','')
        if a not in ['yelp','amazon']:
            dataset_map_list.append(i)
            i+=1
        else:
            if mark<0:
                dataset_map_list.append(i)
                mark = i
                i+=1
            else:
                dataset_map_list.append(mark)

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

def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, epoch,task_id,replay,soft_tgt_batch=None,train_step=None):
    model.train()
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    enable_running_stats(model)
    loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    loss.backward()
    optimizer.first_step(zero_grad=True)

    disable_running_stats(model)
    loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)
    loss.backward()


    optimizer.second_step(zero_grad=False)#

    if task_id !=0 or replay:
        enable_running_stats(model)
        ewc_loss = model.ewc_loss(args.device,task_id,replay)
        if torch.cuda.device_count() > 1:
            ewc_loss = torch.mean(ewc_loss)
        ewc_loss.requires_grad_(True)
        ewc_loss.backward()
        
    optimizer.third_step(zero_grad=True)

    if task_id!=0:
        clamp_params(model,task_id)
    scheduler.step()

    return loss

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


def update_register(model,rho):
    model.register_buffer('pre_rho',rho)

def judge_clamp(n,task_id):
    if "output_layers" not in n:
        return True
    for i in dataset_map_list[:task_id]:
        if  ("."+str(i)+".")  in n and i!=dataset_map_list[task_id]:
            return True
    return False

def clamp_params(model,task_id):
    with torch.no_grad():
        for i,(n, p) in enumerate(model.named_parameters()):
            if p.requires_grad:
                if judge_clamp(n,task_id):
                    n_old = n.replace('.', '__')
                    p_origin = getattr(model, '{}_mean'.format(n_old))
                    bound_value = getattr(model,'pre_rho')
                    p_bound = bound_value*(torch.abs(p_origin))

                    p_upper_bound = p_origin + p_bound 
                    p_lower_bound = p_origin - p_bound 

                    p_upper_mask = p.data > p_upper_bound
                    p_lower_mask = p.data < p_lower_bound
                    p.data[p_upper_mask] = p_upper_bound[p_upper_mask]
                    p.data[p_lower_mask] = p_lower_bound[p_lower_mask]

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    tokenizer_opts(parser)

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")

    parser.add_argument("--adaptive",action='store_true', help="True if using the Adaptive SAM.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--rho", default=0.65 , type=float, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")

    parser.add_argument("--epochs", nargs='+', type=int,
                    default=[10, 10, 10, 10, 10],
                    help='Epoch number for each task')
    parser.add_argument('--n_labeled', type=int, default=2000,
                        help='Number of training data for each class')
    parser.add_argument('--n_val', type=int, default=2000,
                        help='Number of validation data for each class')
    parser.add_argument('--tasks', nargs='+', type=str,
                    default=['ag', 'yelp', 'amazon', 'yahoo', 'dbpedia'],
                    help='Task Sequence')

    parser.add_argument('--select_best', action='store_false',
                    help='whether picking the model with best val acc on each task')
    parser.add_argument('--evaluate_steps', type=int, default=500,
                        help='specific step to evaluate whether current model is the best')

    parser.add_argument("--consolidate",action='store_false',
                help = 'Whether computing the fisher info')
    parser.add_argument("--fisher_estimation_sample_size", type=int, default=1024,
                        help="...")
    parser.add_argument("--lamda", type=int, default=100000,
                        help="coefficient for Find loss")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help='F(i+1)=F_now+gamma*F_old')

    parser.add_argument("--replay", action='store_false',
                        help='replay old samples or not')
    parser.add_argument("--replay_freq", type=int, default=20,
                        help='frequency of replaying, i.e. replay one batch from memory'
                            ' every replay_freq batches')
    parser.add_argument('--kmeans', action='store_false',
                        help='whether applying Kmeans when choosing examples to store')
    parser.add_argument("--store_ratio", type=float, default=0.01,
                        help='how many samples to store for replaying')
    


    adv_opts(parser)

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Get logger.
    args.logger = init_logger(args)

    args.dataset_path_list = []
    for task in args.tasks:
        args.dataset_path_list.append(reduce(os.path.join,sys.path+['datasets']+[task]))

    construct_maplist(args.tasks)#5tasks
    print(dataset_map_list)#5tasks

    args.labels_num_list = [count_labels_num(os.path.join(path, "train.tsv")) for path in args.dataset_path_list]
    args.labels_num_list = remove_dup(args.labels_num_list)

    args.datasets_num = len(args.dataset_path_list)

    args.offset = [0]*len(args.tasks)
    dataset_list = [read_dataset(args, os.path.join(path, "train.tsv"),i) for i,path in enumerate(args.dataset_path_list)]


    train_dataset_list = []
    val_dataset_list = []
    for i,dataset in enumerate(dataset_list):
        train_d, val_d = sample_dataset(args,dataset,dataset_map_list[i])
        train_dataset_list.append(train_d)
        val_dataset_list.append(val_d)
    test_dataset_list = [read_dataset(args, os.path.join(path, "test.tsv"),i) for i,path in enumerate(args.dataset_path_list)]
    for i in range(len(train_dataset_list)):
        args.logger.info("tasks{}:train-{} dev-{} test-{}".format(args.tasks[i],len(train_dataset_list[i]),len(val_dataset_list[i]),len(test_dataset_list[i])))


    model = Classifier(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    batch_size = args.batch_size
    args.logger.info("Batch size: {}".format(batch_size))

    if torch.cuda.device_count() > 1:
        args.logger.info("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    buffer = Memory(args)

    args.logger.info("Start training.")

    pre_rho = torch.tensor(1.0e100)
    args.logger.info("rho: {}".format(args.rho))
    if args.replay:
        args.logger.info("replay on")
    if args.adaptive:
        args.logger.info("adaptive on")

    acc_dic = {}

    for task_id in range(len(args.tasks)):
        if hasattr(model, "module"):
            model.module.change_dataset(dataset_map_list[task_id])
        else:
            model.change_dataset(dataset_map_list[task_id])

        instances_num = len(train_dataset_list[task_id])
        args.train_steps = int(instances_num * args.epochs[task_id] / batch_size) + 1
        args.logger.info("The number of training instances: {} for task {}".format(instances_num,task_id))

        best_model = deepcopy(args.model.state_dict())
        total_loss, result, best_result = 0.0, 0.0, 0.0

        base_optimizer= build_optimizer(args, model)#SAM
        update_register(model,pre_rho)

        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate)

        scheduler = build_sechduler(args,optimizer,task_id)

        for epoch in range(1, args.epochs[task_id] + 1):
            random.shuffle(train_dataset_list[task_id])
            src = torch.LongTensor([example[0] for example in train_dataset_list[task_id]])
            tgt = torch.LongTensor([example[1] for example in train_dataset_list[task_id]])
            seg = torch.LongTensor([example[2] for example in train_dataset_list[task_id]])
            if args.soft_targets:
                soft_tgt = torch.FloatTensor([example[3] for example in train_dataset_list[task_id]])
            else:
                soft_tgt = None

            model.train()
            train_step = 0
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
                if args.replay and  (train_step+1) % args.replay_freq == 0 and task_id > 0:
                    for j in range(task_id):
                        if hasattr(model, "module"):
                            model.module.change_dataset(dataset_map_list[j])
                        else:
                            model.change_dataset(dataset_map_list[j])
                        old_x, old_mask, old_y= buffer.get_random_batch(args.batch_size, j)
                        loss = train_model(args, model, optimizer, scheduler, old_x, old_y, old_mask, epoch,task_id,True,soft_tgt_batch,train_step)
                        total_loss += loss.item()
                    if hasattr(model, "module"):
                        model.module.change_dataset(dataset_map_list[task_id])
                    else:
                        model.change_dataset(dataset_map_list[task_id])
                    loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, epoch,task_id,False,soft_tgt_batch,train_step)
                    total_loss += loss.item()
                else:
                    loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, epoch,task_id,False,soft_tgt_batch,train_step)
                    total_loss += loss.item()

                if (i + 1) % args.report_steps == 0:
                    args.logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                    total_loss = 0.0

                if (train_step+1) % args.evaluate_steps ==0:
                    args.logger.info('---Evaluate---')
                    final_result = 0.0
                    all_len = 0
                    for t_id in range(task_id+1):
                        args.labels_num = args.labels_num_list[dataset_map_list[t_id]]
                        if hasattr(model, "module"):
                            model.module.change_dataset(dataset_map_list[t_id])
                        else:
                            model.change_dataset(dataset_map_list[t_id])
                        t_len = len(val_dataset_list[t_id])
                        result ,c= evaluate(args, val_dataset_list[t_id])
                        final_result +=result*t_len
                        all_len+=t_len
                    final_result /= all_len
                    args.logger.info("Final Acc on all val: {:.4f} ".format(final_result))

                    if final_result > best_result:
                        args.logger.info("------------------Best Model Till Now------------------------")
                        best_result = final_result
                        best_model = deepcopy(model.state_dict())
                train_step+=1


        pre_rho = torch.tensor(args.rho)

        if args.select_best:
            args.model.load_state_dict(deepcopy(best_model))
        args.logger.info('---TEST---')
        final_result = 0.0
        all_len = 0
        for t_id in range(task_id+1):
            args.labels_num = args.labels_num_list[dataset_map_list[t_id]]
            if hasattr(model, "module"):
                model.module.change_dataset(dataset_map_list[t_id])
            else:
                model.change_dataset(dataset_map_list[t_id])
            t_len = len(test_dataset_list[t_id])
            result ,c= evaluate(args, test_dataset_list[t_id])
            final_result +=result*t_len
            all_len+=t_len

            if task_id not in acc_dic:
                acc_dic[task_id] = [str(result*100)]
            else:
                acc_dic[task_id].append(str(result*100))

        final_result /= all_len
        args.logger.info("Final Acc on all test: {:.4f} ".format(final_result))



        if args.replay and task_id < len(train_dataset_list)-1:
            select_samples_to_store(args,model, buffer, train_dataset_list[task_id], task_id)

        cur_task_list = args.tasks[:task_id+1]
        if args.consolidate and task_id < len(train_dataset_list):
            args.logger.info(
                '=> Estimating diagonals of the fisher information matrix...',
            )
            random.shuffle(train_dataset_list[task_id])
            src = torch.LongTensor([example[0] for example in train_dataset_list[task_id]])
            tgt = torch.LongTensor([example[1] for example in train_dataset_list[task_id]])
            seg = torch.LongTensor([example[2] for example in train_dataset_list[task_id]])
            para_fisher = model.estimate_fisher(
                src,tgt,seg, args.fisher_estimation_sample_size,args.batch_size,args.device,args.logger
            )
            model.consolidate(para_fisher,args.gamma)
            args.logger.info(' Done!')
            
    args.logger.info("Summary of ACC")
    for a in acc_dic:
        acc_str = ",".join(acc_dic[a])
        args.logger.info("ACC on Task {}:[{}]".format(a,acc_str))


if __name__ == "__main__":
    main()


