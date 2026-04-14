import os
import math
import random
import datetime
import numpy as np
from scipy.sparse import csr_matrix
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import mindspore as ms

def set_requires_grad(cell, requires_grad):
    for param in cell.get_parameters():
        param.requires_grad = requires_grad
def get_parameter_groups(model):
    group_emb_ebm = list(model.embedding.get_parameters()) + list(model.ebm.get_parameters())
    group_hard = list(model.hard_noise.get_parameters())
    group_soft = list(model.soft_noise.get_parameters())
    group_out = list(model.output_layer.get_parameters())
    return group_emb_ebm, group_hard, group_soft, group_out



class TrainOneStepCell(nn.Cell):
    def __init__(self, network, optimizer):
        super(TrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True)
    def construct(self, *inputs):
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.optimizer.parameters())(*inputs)
        self.optimizer(grads)
        return loss


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur

class EarlyStopping:
    def __init__(self, checkpoint_path, patience=10, verbose=False, delta=0):
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i]+self.delta:
                return False
        return True

    def __call__(self, score, model):
        # score HIT@10 NDCG@10
        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0]*len(score))
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0



def generate_rating_matrix_valid(user_seq, num_users, num_items):
 
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_seq, num_users, num_items):
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def get_rating_matrix(data_name, seq_dic, max_item):
    num_items = max_item + 1
    valid_rating_matrix = generate_rating_matrix_valid(seq_dic['user_seq'], seq_dic['num_users'], num_items)
    test_rating_matrix = generate_rating_matrix_test(seq_dic['user_seq'], seq_dic['num_users'], num_items)

    return valid_rating_matrix, test_rating_matrix

def get_user_seqs_and_max_item(data_file):
    lines = open(data_file).readlines()
    lines = lines[1:]
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split('	', 1)
        items = items.split()
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    return user_seq, max_item



def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT /len(pred_list), NDCG /len(pred_list), MRR /len(pred_list)

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set([actual[i]])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users

def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len([actual[user_id]]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set([actual[user_id]])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res




