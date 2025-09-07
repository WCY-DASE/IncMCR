# Movielens dataset
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn as nn
from utils import print_model_summary
from .Embeddings import *

class MLP_recommendation_module(nn.Module):
    def __init__(self, config):
        super(MLP_recommendation_module, self).__init__()
        self.embedding_dim = config["embedding_dim"]

        # MLP layers
        self.fc1_in_dim = self.embedding_dim * 2 # 用户表示 + 候选 item 表示
        self.fc2_in_dim = config["first_fc_hidden_dim"]  # 预测模型
        self.fc2_out_dim = config["second_fc_hidden_dim"]

        # 构造参数并保存 torch.nn.Parameter(）
        vars = torch.nn.ParameterDict()
        w1 = torch.nn.Parameter(torch.ones([self.fc2_in_dim, self.fc1_in_dim]))  # layer_1
        torch.nn.init.xavier_normal_(w1)  # 参数随机初始化
        vars['recomm_fc_w1'] = w1
        vars['recomm_fc_b1'] = torch.nn.Parameter(torch.zeros(self.fc2_in_dim))

        w2 = torch.nn.Parameter(torch.ones([self.fc2_out_dim, self.fc2_in_dim]))  # layer_2
        torch.nn.init.xavier_normal_(w2)
        vars['recomm_fc_w2'] = w2
        vars['recomm_fc_b2'] = torch.nn.Parameter(torch.zeros(self.fc2_in_dim))

        w3 = torch.nn.Parameter(torch.ones([1, self.fc2_out_dim]))  # output_layer
        torch.nn.init.xavier_normal_(w3)
        vars['recomm_fc_w3'] = w3
        vars['recomm_fc_b3'] = torch.nn.Parameter(torch.zeros(1))

        self.act_f_1 = torch.nn.PReLU()
        self.act_f_2 = torch.nn.PReLU()
        self.vars = vars

    def forward(self, user_emb, item_emb, vars_dict=None):
        if vars_dict is None: # 允许手动指定参数，用作局部更新时暂时赋予fast weights
            vars_dict = self.vars
        x = torch.cat((user_emb, item_emb), 1)
        x = self.act_f_1(F.linear(x, vars_dict['recomm_fc_w1'], vars_dict['recomm_fc_b1']))  # 指定参数，通过神经网络
        x = self.act_f_2(F.linear(x, vars_dict['recomm_fc_w2'], vars_dict['recomm_fc_b2']))
        x = F.linear(x, vars_dict['recomm_fc_w3'], vars_dict['recomm_fc_b3'])  # 评分预测，不能sigmoid
        return x.squeeze()

    def update_parameters(self):  # 获取可更新参数（local update）
        return self.vars

class Transformation_module(nn.Module):
    def __init__(self, config, input_dim):
        super(Transformation_module, self).__init__()
        self.embedding_dim = config["embedding_dim"]

        # MLP based transformation
        self.fc1_in_dim = input_dim
        self.fc1_out_dim = self.embedding_dim

        vars = torch.nn.ParameterDict()
        w1 = torch.nn.Parameter(torch.ones([self.fc1_out_dim, self.fc1_in_dim]))  # layer_1
        torch.nn.init.xavier_normal_(w1)  # 参数随机初始化
        vars['transform_fc_w1'] = w1
        vars['transform_fc_b1'] = torch.nn.Parameter(torch.zeros(self.fc1_out_dim))

        self.act_f = torch.nn.PReLU()
        self.vars = vars

    def forward(self, emb, vars_dict=None):
        if vars_dict is None:  # 允许手动指定参数，用作局部更新时暂时赋予fast weights
            vars_dict = self.vars
        x = self.act_f(F.linear(emb, vars_dict['transform_fc_w1'], vars_dict['transform_fc_b1']))  # 指定参数，通过神经网络
        return x

    def update_parameters(self):  # 获取可更新参数（local update）
        return self.vars

class TwoTower_Recommender(nn.Module):
    def __init__(self, config):
        super(TwoTower_Recommender, self).__init__()
        self.config = config
        self.dataset = config["dataset"]
        self.embedding_dim = config["embedding_dim"]

        # # Meta_initial_Emb (这部分可以扩展)
        # self.meta_initial_userID_emb = torch.nn.Parameter(torch.ones([1,self.embedding_dim]))  # meta_initial_user_emb
        # torch.nn.init.uniform_(self.meta_initial_userID_emb)  # 参数随机初始化

        # user & item Embedding
        if self.dataset == 'Movielens10M':
            self.item_emb = ItemEmbeddingML(config)
            self.user_emb = UserEmbeddingML(config)

        # user & item transformation tower
        self.user_transform = Transformation_module(config, config["user_embedding_dim"])  # user representation (固定维度)
        self.item_transform = Transformation_module(config, config["item_embedding_dim"])  # item representation (固定维度)

        # recommendation module
        self.recomm_module = MLP_recommendation_module(config)

        print_model_summary(self,"TwoTower Model")

    def forward(self, user, item, user_trans_dict=None, item_trans_dict=None, recomm_dict=None):
        # user_embs = self.get_meta_user_embs(user.shape[0], user_initial, adapted_u_ids)
        user_emb = self.user_emb(user)
        x_u = self.user_transform(user_emb, user_trans_dict)   #  user repre
        item_emb = self.item_emb(item)
        x_i = self.item_transform(item_emb, item_trans_dict) #  item repre

        x = self.recomm_module(x_u, x_i, recomm_dict)
        return x

    # def get_meta_user_embs(self, interaction_num, user_initial, adapted_id_emb):
    #     if user_initial:
    #         user_embs = self.meta_initial_userID_emb.repeat((interaction_num,1))
    #     else:
    #         user_embs = adapted_id_emb.repeat((interaction_num,1))
    #     return user_embs









