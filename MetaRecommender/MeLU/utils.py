import os
import numpy as np
import time
import torch
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

def load_pickle_data(data_file):
    fr = open(data_file, "rb")
    result = pickle.load(fr)
    return result

def check_and_create_path(filename):
    file_dir = os.path.split(filename)[0]
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

def init_seed(seed=None): # 设置numpy以及torch的随机种子
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def print_model_summary(model, model_name):
    print("\nParameters for {}!".format(model_name))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name,param.shape)
        else:
            print(name, param.shape, "[no grads]")


class Evaluation:
    def __init__(self):
        self.k = 5

    def prediction(self, real_score, pred_score):
        MAE = mean_absolute_error(real_score, pred_score)
        RMSE = math.sqrt(mean_squared_error(real_score, pred_score))
        return MAE, RMSE

    def dcg_at_k(self,scores):
        # assert scores
        return scores[0] + sum(sc / math.log(ind+1, 2) for sc, ind in zip(scores[1:], range(2, len(scores) + 1)))   # 第i个位置的效益为math.log(ind+1, 2)

    def ranking(self, real_score, pred_score, k):
        # NDCG@K
        pred_sorted_idx = np.argsort(pred_score)[::-1][:k] # 按预测取top-K
        p_s_at_k = real_score[pred_sorted_idx] # pred_list
        pred_dcg = self.dcg_at_k(p_s_at_k) # 按 预测排序结果，用真实分数计算ncdg

        real_sorted_idx = np.argsort(real_score)[::-1][:k]  # 按预测取top-K
        r_s_at_k = real_score[real_sorted_idx]  # real_list
        idcg = self.dcg_at_k(r_s_at_k)
        return pred_dcg/idcg
