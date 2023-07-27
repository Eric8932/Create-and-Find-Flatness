#!/usr/bin/env python

"""
This script provides an example to wrap UER-py for multi-task classification.
"""
from logging.config import valid_ident
import sys
import os
import random
import argparse
import torch
import torch.nn as nn
import numpy as np
from functools import reduce
from copy import deepcopy

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.embeddings import *
from uer.encoders import *
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.utils.logging import init_logger
from uer.utils.misc import pooling
from uer.model_saver import save_model
from uer.opts import *
from finetune.run_classifier import count_labels_num, evaluate, build_optimizer, load_or_initialize_parameters,train_model

dataset_map_list = []

def compute_class_offsets(tasks, task_classes):
    '''
    :param tasks: a list of the names of tasks, e.g. ["amazon", "yahoo"]
    :param task_classes:  the corresponding numbers of classes, e.g. [5, 10]
    :return: the class # offsets, e.g. [0, 5]
    Here we merge the labels of yelp and amazon, i.e. the class # offsets
    for ["amazon", "yahoo", "yelp"] will be [0, 5, 0]
    '''
    task_num = len(tasks)
    offsets = [0] * task_num
    prev = -1
    total_classes = 0
    for i in range(task_num):
        if tasks[i] in ["amazon", "yelp"]:
            if prev == -1:
                prev = i
                offsets[i] = total_classes
                total_classes += task_classes[i]
            else:
                offsets[i] = offsets[prev]
        else:
            offsets[i] = total_classes
            total_classes += task_classes[i]
    return total_classes, offsets

class MultitaskClassifier(nn.Module):
    def __init__(self, args):
        super(MultitaskClassifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.pooling_type = args.pooling
        self.preknow_task = args.preknow_task

        if args.preknow_task:
            self.output_layers_1 = nn.Linear(args.hidden_size, args.hidden_size)
            self.output_layers_2 = nn.Linear(args.hidden_size, args.labels_num)
        else:
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

        if self.preknow_task:
            output = torch.tanh(self.output_layers_1(output))
            logits = self.output_layers_2(output)
        else:
            output = torch.tanh(self.output_layers_1[self.dataset_id](output))
            logits = self.output_layers_2[self.dataset_id](output)

        if tgt is not None:
            loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits

    def change_dataset(self, dataset_id):
        self.dataset_id = dataset_id

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

def sample_dataset(args, dataset,i):#dataset[(src,tgt,seg)]
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
        # idxs = np.array(train_df[train_df[0] == cls].index)
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


def pack_dataset(dataset, dataset_id, batch_size):
    packed_dataset = []
    src_batch, tgt_batch, seg_batch = [], [], []
    for i, sample in enumerate(dataset):
        src_batch.append(sample[0])
        tgt_batch.append(sample[1])
        seg_batch.append(sample[2])
        if (i + 1) % batch_size == 0:
            packed_dataset.append((dataset_id, torch.LongTensor(src_batch), torch.LongTensor(tgt_batch), torch.LongTensor(seg_batch)))
            src_batch, tgt_batch, seg_batch = [], [], []
            continue
    if len(src_batch) > 0:
        packed_dataset.append((dataset_id, torch.LongTensor(src_batch), torch.LongTensor(tgt_batch), torch.LongTensor(seg_batch)))

    return packed_dataset

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

def remove_dup(l_list):
    l = []
    for a in l_list:
        if a not in l or a!=5:
            l.append(a)
    return l



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    # parser.add_argument("--dataset_path_list", default=[], nargs='+', type=str, help="Dataset path list.")
    parser.add_argument('--tasks', nargs='+', type=str,
                    default=['ag', 'yelp', 'amazon', 'yahoo', 'dbpedia'],
                    help='Task Sequence')

    parser.add_argument("--output_model_path", default="models/multitask_classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--config_path", default="models/bert/base_config.json", type=str,
                        help="Path of the config file.")

    parser.add_argument('--n_labeled', type=int, default=2000,
                        help='Number of training data for each class')
    parser.add_argument('--n_val', type=int, default=2000,
                        help='Number of validation data for each class')
    parser.add_argument('--select_best', action='store_false',
                    help='whether picking the model with best val acc on each task')
    parser.add_argument('--preknow_task', action='store_true',
                        help='Whether knowing all tasks before training. Deciding the construction of the task layer')
    parser.add_argument('--evaluate_steps', type=int, default=800,
                        help='specific step to evaluate whether current model is the best')


    # Model options.
    model_opts(parser)

    # Tokenizer options.
    tokenizer_opts(parser)

    # Optimizer options.
    optimization_opts(parser)

    # Training options.
    training_opts(parser)

    adv_opts(parser)

    args = parser.parse_args()

    args.soft_targets = False

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    args.dataset_path_list = []
    for task in args.tasks:
        args.dataset_path_list.append(reduce(os.path.join,sys.path+['datasets']+[task]))

    
    construct_maplist(args.tasks)
    print(dataset_map_list)
    # Count the number of labels.
    args.labels_num_list = [count_labels_num(os.path.join(path, "train.tsv")) for path in args.dataset_path_list]
    args.labels_num_list = remove_dup(args.labels_num_list)

    args.datasets_num = len(args.dataset_path_list)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Get logger.
    args.logger = init_logger(args)

    # Training phase.
    if args.preknow_task:
        total_classes,offset = compute_class_offsets(args.tasks,args.labels_num_list)
        args.labels_num = total_classes
        args.offset = offset
    else:
        args.offset = [0]*len(args.tasks)
    dataset_list = [read_dataset(args, os.path.join(path, "train.tsv"),i) for i,path in enumerate(args.dataset_path_list)]
    print('READ DATASET DONE')
    print(len(dataset_list))

    train_dataset_list = []
    val_dataset_list = []
    for i,dataset in enumerate(dataset_list):
        train_d, val_d = sample_dataset(args,dataset,dataset_map_list[i])
        # print(len(train_d))
        train_dataset_list.append(train_d)
        val_dataset_list.append(val_d)

    packed_dataset_list = [pack_dataset(dataset, i, args.batch_size) for i, dataset in enumerate(train_dataset_list)]
    packed_dataset_all = []
    for packed_dataset in packed_dataset_list:
        packed_dataset_all += packed_dataset

    test_dataset_list = [read_dataset(args, os.path.join(path, "test.tsv"),i) for i,path in enumerate(args.dataset_path_list)]
    for i in range(len(train_dataset_list)):
        args.logger.info("tasks{}:train-{} dev-{} test-{}".format(args.tasks[i],len(train_dataset_list[i]),len(val_dataset_list[i]),len(test_dataset_list[i])))


    instances_num = sum([len(dataset) for dataset in train_dataset_list])
    batch_size = args.batch_size

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    args.logger.info("Batch size: {}".format(batch_size))
    args.logger.info("The number of training instances: {}".format(instances_num))

    # Build multi-task classification model.
    model = MultitaskClassifier(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.device = 'mps'
    model = model.to(args.device)
    args.model = model

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        args.logger.info("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    best_model = deepcopy(model.state_dict())
    total_loss, result, best_result = 0.0, 0.0, 0.0

    args.logger.info("Start training.")

    train_step = 0

    acc_dic = {}

    for epoch in range(1, args.epochs_num + 1):
        random.shuffle(packed_dataset_all)
        model.train()
        for i, (dataset_id, src_batch, tgt_batch, seg_batch) in enumerate(packed_dataset_all):
            if not args.preknow_task:
                if hasattr(model, "module"):
                    model.module.change_dataset(dataset_map_list[dataset_id])
                else:
                    model.change_dataset(dataset_map_list[dataset_id])
            model.train()
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, None)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                args.logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

            if (train_step+1) % args.evaluate_steps ==0:
                args.logger.info('---Evaluate---')
                final_result = 0.0
                all_len = 0
                if not args.preknow_task:
                    for t_id in range(len(args.tasks)):
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
                else:
                    temp_val_list = deepcopy(val_dataset_list[0])
                    for t_id in range(1,len(args.tasks)):
                        temp_val_list+= val_dataset_list[t_id]
                    final_result ,c= evaluate(args, temp_val_list)

                if final_result > best_result:
                    args.logger.info("------------------Best Model Till Now------------------------")
                    best_result = final_result
                    best_model = deepcopy(model.state_dict())
                    # best_predictor = deepcopy(predictor.state_dict())
            train_step+=1


    if args.select_best:
        args.model.load_state_dict(deepcopy(best_model))

    final_result = 0.0
    all_len = 0
    if not args.preknow_task:
        for t_id in range(len(args.tasks)):
            args.labels_num = args.labels_num_list[dataset_map_list[t_id]]
            if hasattr(model, "module"):
                model.module.change_dataset(dataset_map_list[t_id])
            else:
                model.change_dataset(dataset_map_list[t_id])
            t_len = len(test_dataset_list[t_id])
            result ,c= evaluate(args, test_dataset_list[t_id])
            final_result +=result*t_len
            all_len+=t_len

            if len(args.tasks) not in acc_dic:
                acc_dic[len(args.tasks)] = [str(result*100)]
            else:
                acc_dic[len(args.tasks)].append(str(result*100))

        final_result /= all_len
        args.logger.info("Final Acc on all test: {:.4f} ".format(final_result))
    else:
        temp_test_list = deepcopy(test_dataset_list[0])
        for t_id in range(1,len(args.tasks)):
            temp_test_list+= test_dataset_list[t_id]
        final_result ,c= evaluate(args, temp_test_list)

    args.logger.info("Summary of ACC")
    for a in acc_dic:
        acc_str = ",".join(acc_dic[a])
        args.logger.info("ACC on Task {}:[{}]".format(a,acc_str))

if __name__ == "__main__":
    main()
