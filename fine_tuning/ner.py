# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
import argparse
import torch
import pickle
sys.path.append("/data/pengjie.ding/code/XLM")

from src.utils import AttrDict
from src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from src.utils import bool_flag, initialize_exp
from fine_tuning.model import XLM_BiLSTM_CRF
from fine_tuning.data_utils import generate_DataLoader
# from fine_tuning.train_fc import train
from fine_tuning.train_lstm_crf import train

# parse parameters
parser = argparse.ArgumentParser(description='fine-tuning downstream task: ner')

# main parameters
parser.add_argument("--exp_name", type=str, default="CCKS2019",
                    help="Experiment name")
parser.add_argument("--dump_path", type=str, default="dump_ner",
                    help="Experiment dump path")
parser.add_argument("--exp_id", type=str, default="",
                    help="Experiment ID")

parser.add_argument("--model_path", type=str, default="../model/mlm_tlm_xnli15_1024.pth",
                    help="Model location")

# data
parser.add_argument("--data_path", type=str, default="",
                    help="Data path")
parser.add_argument("--num_class", type=int, default=15,
                    help="Num of classes")

# batch parameters
parser.add_argument("--max_len", type=int, default=100,
                    help="Maximum length of sentences (after BPE)")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Number of sentences per batch")
parser.add_argument("--tokens_per_batch", type=int, default=-1,
                    help="Number of tokens per batch")

# model / optimization
parser.add_argument("--finetune_layers", type=str, default='0:_1',
                    help="Layers to finetune. 0 = embeddings, _1 = last encoder layer")
parser.add_argument("--weighted_training", type=bool_flag, default=False,
                    help="Use a weighted loss during training")
parser.add_argument("--dropout", type=float, default=0.5,
                    help="Fine-tuning dropout")
parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001",
                    help="Optimizer")
parser.add_argument("--n_epochs", type=int, default=100,
                    help="Maximum number of epochs")
parser.add_argument("--epoch_size", type=int, default=-1,
                    help="Epoch size (-1 for full pass over the dataset)")
parser.add_argument("--initializer_range", type=float, default=0.02,
                    help="Initializer range for init weights")

# lstm parameters
parser.add_argument("--embedding_dim", type=int, default=1024,
                    help="Embedding dim of LSTM")
parser.add_argument("--hidden_dim", type=int, default=200,
                    help="Hidden dim of LSTM")


# parse parameters
args = parser.parse_args()
if args.tokens_per_batch > -1:
    args.group_by_size = True


# reload pretrained model
reloaded = torch.load(args.model_path)
params = AttrDict(reloaded['params'])
dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
params.n_words = len(dico)
params.bos_index = dico.index(BOS_WORD)
params.eos_index = dico.index(EOS_WORD)
params.pad_index = dico.index(PAD_WORD)
params.unk_index = dico.index(UNK_WORD)
params.mask_index = dico.index(MASK_WORD)


# load training dataset
with open('../model/dataset_seg/train/train.pkl', 'rb') as inp:
    x_train = pickle.load(inp)
    y_train = pickle.load(inp)
    x_test = pickle.load(inp)
    y_test = pickle.load(inp)
    tag2id = pickle.load(inp)
    id2tag = pickle.load(inp)
print("train len:", len(x_train))
print("test len:", len(x_test))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_iter, dev_iter = generate_DataLoader(x_train, x_test, y_train, y_test, args.batch_size)

model = XLM_BiLSTM_CRF(args, len(tag2id), params, dico, reloaded).to(device)
train(model, train_iter, dev_iter, args)
