# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch import cuda
import torch.nn.functional as F
from transformers import logging
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

sys.path.append("")
from classifier.mbert_train import SCIterator

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = 'cuda' if cuda.is_available() else 'cpu'


def main():
    parser = argparse.ArgumentParser('Evaluating Style Strength')
    parser.add_argument('-order', default=0, type=str, help='the training order')
    parser.add_argument('-style', default=0, type=int, help='get informal (0) or formal (1) scores')
    parser.add_argument('-max_len', default=40, type=int, help='max tokens in a batch')
    parser.add_argument('-lang', default='en_XX', type=str, help='the name of language')
    parser.add_argument('-seed', default=42, type=int, help='pseudo random number seed')
    parser.add_argument('-batch_size', default=128, type=int, help='max sents in a batch')
    parser.add_argument('-dataset', default='xformal', type=str, help='the dataset name')
    parser.add_argument('-output', default='mbart_gyafc_vanilla.txt', type=str, help='the output file name')

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased")
    model_dir = '/data/s3238903/dutch-style-transfer/mbert_mformal.chkpt'.format(opt.lang)
    model.load_state_dict(torch.load(model_dir))
    model.to(device).eval()

    test_src, test_tgt = [], []
    with open('outputs/{}'.format(opt.output),'r') as f:
        for line in f.readlines():
            test_tgt.append(tokenizer.encode(line.strip()))

    print('[Info] {} instances from src test set'.format(len(test_src)))
    print('[Info] {} instances from tgt test set'.format(len(test_tgt)))
    test_loader = SCIterator(test_src, test_tgt, tokenizer.pad_token_id, False, opt)

    corre_num = 0.
    total_num = 0.
    loss_list = []
    with torch.no_grad():
        for i,batch in enumerate(test_loader):
            src, tgt = map(lambda x: x.to(device), batch)
            mask = src.ne(tokenizer.pad_token_id).long()
            outs = model(src, mask, labels=tgt)
            loss, logits = outs[:2]
            soft_max = nn.Softmax(dim=1)
            normalised_scores = soft_max(logits)
            #y_hat = logits.argmax(dim=-1)
            #same = [int(p == q) for p, q in zip(tgt, y_hat)]
            #corre_num += sum(same)
            #total_num += len(tgt)
            #loss_list.append(loss.item())

    #print('[Info] Test: {}'.format('acc {:.2f}% | loss {:.4f}').format(corre_num / total_num * 100, np.mean(loss_list)))
    # get scores on formality
    if opt.style == 1:
        formal_scores = [i[1].item() for i in normalised_scores]
        model_style_score = np.mean(formal_scores)
        print(formal_scores, "\nstyle {} strength (mbert classifier): {}".format(opt.style, model_style_score))
    else:
        formal_scores = [i[0].item() for i in normalised_scores]
        model_style_score = np.mean(formal_scores)
        print(formal_scores, "\nstyle {} strength (mbert classifier): {}".format(opt.style, model_style_score))


if __name__ == '__main__':
    main()
