"""
This script provides an example to wrap UER-py for sequential classification with EWC loss.
"""
import sys
import os
import random
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
from functools import reduce
import numpy as np
from copy import deepcopy

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
from finetune.run_cls_mt import read_dataset,sample_dataset,remove_dup
from finetune.run_cls_cf import Memory, select_samples_to_store

dataset_map_list = []#5tasks

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.lamda = args.lamda

        self.pooling_type = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layers_1 = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in args.labels_num_list])
        self.output_layers_2 = nn.ModuleList([nn.Linear(args.hidden_size, labels_num) for labels_num in args.labels_num_list])
        self.dataset_id = 0


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

    def compute_features(self, src,seg):
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
        

        return output

    def estimate_fisher(self, src,tgt,seg, sample_size, batch_size,device,logger):
        # sample loglikelihoods from the dataset.

        loglikelihood_grads = []
        for (src_batch,tgt_batch,seg_batch,_) in batch_loader(batch_size,src,tgt,seg):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            seg_batch = seg_batch.to(device)
            logits_l = F.log_softmax(self.compute_logits(src_batch,seg_batch), dim=1)[range(batch_size), tgt_batch]

            for i,l in enumerate(logits_l,1):
                grad_l = []
                gra = autograd.grad(l, self.parameters(),retain_graph= (i<len(logits_l)),allow_unused=True)
                # gra = gra.detach().cpu()
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

    def consolidate(self, fisher,gamma):
        for n, p in self.named_parameters():
            if "output_layers" not in n or ("."+str(self.dataset_id)+".") in n:
                n = n.replace('.', '__')
                self.register_buffer('{}_mean'.format(n), p.data.clone())
                if hasattr(self,'{}_fisher'.format(n)):
                    fisher_old = getattr(self, '{}_fisher'.format(n))
                    fisher_new = fisher_old*gamma + fisher[n].data.clone()
                    self.register_buffer('{}_fisher'
                                    .format(n), fisher_new)
                else:
                    self.register_buffer('{}_fisher'
                                    .format(n), fisher[n].data.clone())

    def ewc_loss(self, device,task_id,replay):
        try:
            losses = []
            for n, p in self.named_parameters():
                if "output_layers" not in n or (self.dataset_id<task_id and self.dataset_id!=dataset_map_list[task_id] and ("."+str(self.dataset_id)+".") in n):
                    n = n.replace('.', '__')
                    mean = getattr(self, '{}_mean'.format(n))
                    fisher = getattr(self, '{}_fisher'.format(n))
                    # wrap mean and fisher in variables.
                    mean = Variable(mean)
                    fisher = Variable(fisher)
                    # calculate a ewc loss. (assumes the parameter's prior as
                    # gaussian distribution with the estimated mean and the
                    # estimated cramer-rao lower bound variance, which is
                    # equivalent to the inverse of fisher information)
                    losses.append((fisher * (p-mean)**2).sum())
            return (self.lamda/2)*sum(losses).to(device)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1)).to(device)
            )

def construct_maplist(tasks):
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


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler


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

def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, task_id,replay,soft_tgt_batch=None,train_step=None):
    model.train()
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)

    ewc_loss = model.ewc_loss(args.device,task_id,replay)

    loss = loss+ewc_loss
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


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


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    tokenizer_opts(parser)

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")
    parser.add_argument('--tasks', nargs='+', type=str,
                default=['ag', 'yelp', 'amazon', 'yahoo', 'dbpedia'],
                help='Task Sequence')
    parser.add_argument('--n_labeled', type=int, default=2000,
                        help='Number of training data for each class')
    parser.add_argument('--n_val', type=int, default=2000,
                        help='Number of validation data for each class')
    parser.add_argument("--epochs", nargs='+', type=int,
                    default=[10, 10, 10, 10, 10],
                    help='Epoch number for each task')
    parser.add_argument("--consolidate",action='store_false',
                help = 'Whether computing the fisher info')
    parser.add_argument("--fisher_estimation_sample_size", type=int, default=1024,
                        help="...")
    parser.add_argument("--lamda", type=int, default=100000,
                        help="coefficient for ewc_loss")
    parser.add_argument("--gamma", type=float, default=0,
                        help='F(i+1)=F_now+gamma*F_old')

    parser.add_argument('--select_best', action='store_false',
                    help='whether picking the model with best val acc on each task')
    parser.add_argument('--evaluate_steps', type=int, default=500,
                        help='specific step to evaluate whether current model is the best')

    parser.add_argument("--replay", action='store_false',
                        help='replay old samples or not')
    parser.add_argument("--replay_freq", type=int, default=20,
                        help='frequency of replaying, i.e. replay one batch from memory'
                            ' every replay_freq batches')
    parser.add_argument('--kmeans', action='store_false',
                        help='whether applying Kmeans when choosing examples to store')
    parser.add_argument("--store_ratio", type=float, default=0.01,
                        help='how many samples to store for replaying')

    parser.add_argument('--save_model_path',default=None,help='save the model after each task')
    parser.add_argument('--save_fisher_path',default=None,help='save fisher info for current model after each task')



    adv_opts(parser)

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    set_seed(args.seed)

    args.logger = init_logger(args)

    args.dataset_path_list = []
    for task in args.tasks:
        args.dataset_path_list.append(reduce(os.path.join,sys.path+['datasets']+[task]))
        
    construct_maplist(args.tasks)
    print(dataset_map_list)

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

    # Build classification model.
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
    if args.replay:
        args.logger.info("replay on")

    acc_dic = {}

    for task_id in range(len(args.tasks)):
        if hasattr(model, "module"):
            model.module.change_dataset(dataset_map_list[task_id])
        else:
            model.change_dataset(dataset_map_list[task_id])

        instances_num = len(train_dataset_list[task_id])

        args.train_steps = int(instances_num * args.epochs[task_id] / batch_size) + 1
        args.logger.info("The number of training instances: {} for task {}".format(instances_num,task_id))

        optimizer, scheduler = build_optimizer(args, model)
        best_model = deepcopy(args.model.state_dict())
        total_loss, result, best_result = 0.0, 0.0, 0.0

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
                        loss = train_model(args, model, optimizer, scheduler, old_x, old_y, old_mask,task_id,True,soft_tgt_batch,train_step)
                        total_loss += loss.item()
                    if hasattr(model, "module"):
                        model.module.change_dataset(dataset_map_list[task_id])
                    else:
                        model.change_dataset(dataset_map_list[task_id])
                    loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch,task_id,False,soft_tgt_batch,train_step)
                    total_loss += loss.item()
                else:
                    loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch,task_id,False,soft_tgt_batch,train_step)
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


        if args.replay:
            select_samples_to_store(args,model, buffer, train_dataset_list[task_id], task_id)

        if args.consolidate and task_id < len(train_dataset_list):
            # estimate the fisher information of the parameters and consolidate
            # them in the network.
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


