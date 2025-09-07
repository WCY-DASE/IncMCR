import torch
import numpy as np
# from BaseModel import UserEmbeddingML, ItemEmbeddingML,ItemEmbeddingDB,UserEmbeddingDB,ItemEmbeddingYP,UserEmbeddingYP,NCF_RecommModule
from .BaseModel import TwoTower_Recommender
from utils import print_model_summary,Evaluation
import collections

class KnowledgeDistllationLoss(torch.nn.Module):
    def __init__(self, temperature = 1):
        super(KnowledgeDistllationLoss, self).__init__()
        self.temperature = temperature
        self.KD_loss_function = torch.nn.MSELoss()

    def forward(self, teacher_outputs, student_outputs):
        KD_loss = self.KD_loss_function(teacher_outputs, student_outputs)/ (2 * self.temperature ** 2)
        return KD_loss


class Melu(torch.nn.Module):
    def __init__(self, config, model_name, stage):
        super(Melu, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if config['use_cuda'] else "cpu")
        self.model_name = model_name
        # self.input_dir = config["Input_dir"]

        ### Base Model
        # Two Tower Recommenders
        self.rec_model = TwoTower_Recommender(config)  # 在每个任务基于meta-learning 进行local adaptation
        print_model_summary(self, self.model_name)

        ### Meta learner
        #  Base model's meta-learned parameters
        print("Meta-learned Parameters of Base model {}!".format(config["base_model"]))
        # PART 1: Recommendation module
        if "recomm" in config['Meta_learned_para']:
            print("Recommendation Module:--------------------")
            for name, param in self.rec_model.recomm_module.update_parameters().items():
                print(name, param.shape)
        # PART 2: User transformation module
        if "user_transform" in config['Meta_learned_para']:
            print("User Transformation Module:--------------------")
            for name, param in self.rec_model.user_transform.update_parameters().items():
                print(name, param.shape)
        # PART 3: Item transformation module
        if "item_transform" in config['Meta_learned_para']:
            print("Item Transformation Module:--------------------")
            for name, param in self.rec_model.item_transform.update_parameters().items():
                print(name, param.shape)

        # 加载预训练模型
        if stage == "Pretrain":
            print('---------------Load pretrained model from {} ---------------'.format(self.config["pretrained_modelfile"]))
            self.load_state_dict(torch.load(self.config["pretrained_modelfile"], map_location=self.device))

        # Meta-Training settting
        self.global_lr = config['global_lr']  # 学习率
        self.local_lr = config['local_lr']
        self.lr_decay_step = config['lr_ml_step']
        self.lr_decay_ratio = config['lr_dc']

        # Incremental setup
        self.KD_loss_ratio = config['KD_loss_ratio']

        self.loss_function = torch.nn.MSELoss()
        self.KD_loss_function = KnowledgeDistllationLoss()

        # support loss optimizer (在 全部samples上 进行一阶更新)
        sample_para_list = [{"params": self.rec_model.user_emb.parameters()}, {"params": self.rec_model.item_emb.parameters()}]
        self.local_optimizer = torch.optim.Adam(sample_para_list, lr=self.local_lr)  # 包含base model的初始化参数，以及meta-learner的可学习参数
        print("\nParameters for local optimizer!")
        for param_group in self.local_optimizer.param_groups:
            print("-------------------------")
            for param in param_group["params"]:
                print(param.name, param.shape)

        #  meta-loss optimizer (在query samples上 进行二阶更新)
        meta_para_list = [{"params": self.rec_model.user_transform.parameters()}, {"params": self.rec_model.item_transform.parameters()}, {"params": self.rec_model.recomm_module.parameters()}]
        self.meta_optimizer = torch.optim.Adam(meta_para_list, lr=self.global_lr)
        print("\nParameters for Meta optimizer!")
        for param_group in self.meta_optimizer.param_groups:
            print("-------------------------")
            for param in param_group["params"]:
                print(param.name, param.shape)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.meta_optimizer, milestones=self.lr_decay_step, gamma=self.lr_decay_ratio)  # version_2: MultiStepLR

        self.cal_metrics = Evaluation()

    def global_update(self, train_tasks, device=torch.device('cpu'), for_replay = False):
        """
            每个task进行local update, batch上global update
        """
        task_ids, support_xs_u, support_xs_v, support_ys, query_xs_u, query_xs_v, query_ys = zip(*train_tasks)
        batch_sz = len(support_xs_u)
        meta_loss_s = []
        # meta_grads_s = []
        mae_s, rmse_s, ndcg_at_3_s, ndcg_at_5_s, ndcg_at_10_s= [],[],[],[],[]

        for i in range(batch_sz):  # each task in a batch
            # Local Update
            # print("Task ", task_ids[i])

            _loss, _mae, _rmse, _ndcg_at_3, _ndcg_at_5, _ndcg_at_10 = self.local_update(support_xs_u[i], support_xs_v[i],
                                                                                    support_ys[i], query_xs_u[i], query_xs_v[i],
                                                                                    query_ys[i], mode="train", device=device, for_replay = for_replay) # 对每个任务进行local update, 并返回query loss
            meta_loss_s.append(_loss)
            mae_s.append(_mae)
            rmse_s.append(_rmse)
            ndcg_at_3_s.append(_ndcg_at_3)
            ndcg_at_5_s.append(_ndcg_at_5)
            ndcg_at_10_s.append(_ndcg_at_10)

        meta_loss = torch.stack(meta_loss_s).mean(0)  # 得到batch上的query loss 的平均
        mae = np.mean(mae_s)
        rmse = np.mean(rmse_s)
        ndcg_at_3 = np.mean(ndcg_at_3_s)
        ndcg_at_5 = np.mean(ndcg_at_5_s)
        ndcg_at_10 = np.mean(ndcg_at_10_s)

        # print(">>>>>>>>>>>>>>>Global Update>>>>>>>>>>>>>>>>>")
        # Global update
        # self.meta_optimizer.zero_grad() # 对每个batch调用，将之前梯度置为0
        # Total_loss = meta_loss
        # Total_loss.backward()  # loss 反向传播进行梯度计算
        # self.meta_optimizer.step() # 进行一步优化，只有这一步才真正进行了参数的优化，之前的local_update都是手动进行更新并保存，并手动赋值更新后的参数。
        return meta_loss.item(), mae, rmse, ndcg_at_3, ndcg_at_5, ndcg_at_10  # 每个batch上的query set上的计算结果

    def local_update(self,support_set_x_u, support_set_x_v, support_set_y, query_set_x_u, query_set_x_v, query_set_y, mode="train",device=torch.device('cpu'), for_replay = False):

        # trans array to tensord
        support_set_x_u = torch.tensor(support_set_x_u).to(device)
        support_set_x_v = torch.tensor(support_set_x_v).to(device)
        support_set_y = torch.tensor(support_set_y).to(device)
        query_set_x_u = torch.tensor(query_set_x_u).to(device)
        query_set_x_v = torch.tensor(query_set_x_v).to(device)
        query_set_y = torch.tensor(query_set_y).to(device)

        ### 保留base model (recommendation module)初始参数
        # rec_initial_weights = collections.OrderedDict(self.rec_model.recomm_module.update_parameters())  # 预测模块
        # user_trans_initial_weights = collections.OrderedDict(self.rec_model.user_transform.update_parameters())
        # item_trans_initial_weights = collections.OrderedDict(self.rec_model.item_transform.update_parameters())
        # #
        # # ### Local Adaptation on Suppport set
        # support_set_y_pred = self.rec_model(support_set_x_u, support_set_x_v)  # 预测
        # loss = self.loss_function(support_set_y_pred.to(torch.float32), support_set_y.to(torch.float32))  # torch.nn.functional.mse_loss 计算loss
        # grad_rec = torch.autograd.grad(loss, rec_initial_weights.values(), retain_graph=True)  # 指定参数（recomm_module）,计算梯度 (rtrain为保留计算图，为后续其他参数求导方便)(create为梯度保留计算图，为求导高阶参数)
        # grad_user_trans = torch.autograd.grad(loss, user_trans_initial_weights.values(), retain_graph=True)  # 指定参数（User trandormation tower）,计算梯度
        # grad_item_trans = torch.autograd.grad(loss, item_trans_initial_weights.values(), retain_graph=True)  # 指定参数（Item trandormation tower）,计算梯度
        #
        # ### Local Update (手动更新 fast weights + 自动更新 embeddings)
        #
        # rec_fast_weights = collections.OrderedDict()
        # user_trans_fast_weights = collections.OrderedDict()
        # item_trans_fast_weights = collections.OrderedDict()
        #
        # for para_i, para_name in enumerate(rec_initial_weights.keys()):
        #     rec_fast_weights[para_name] = rec_initial_weights[para_name] - self.local_lr * grad_rec[para_i]
        # for para_i, para_name in enumerate(user_trans_initial_weights.keys()):
        #     user_trans_fast_weights[para_name] = user_trans_initial_weights[para_name] - self.local_lr * grad_user_trans[para_i]
        # for para_i, para_name in enumerate(item_trans_initial_weights.keys()):
        #     item_trans_fast_weights[para_name] = item_trans_initial_weights[para_name] - self.local_lr * grad_item_trans[para_i]
        #
        # # 自动更新
        # self.support_optimizer.zero_grad()  # 将之前梯度置为0d
        # grad_user_emb = torch.autograd.grad(loss, self.rec_model.user_emb.parameters(),retain_graph=True) # loss 反向传播进行梯度计算
        # grad_item_emb = torch.autograd.grad(loss, self.rec_model.item_emb.parameters()) # loss 反向传播进行梯度计算
        # self.support_optimizer.step()

        ### 对replay tasks, 保留meta-model的outputs, 避免发生偏移
        if for_replay:
            # teacher logits
            meta_support_logits = self.rec_model(support_set_x_u, support_set_x_v).detach() # 不从teacher model处计算梯度,

        rec_fast_weights = collections.OrderedDict(self.rec_model.recomm_module.update_parameters())  # 预测模块
        user_trans_fast_weights = collections.OrderedDict(self.rec_model.user_transform.update_parameters())
        item_trans_fast_weights = collections.OrderedDict(self.rec_model.item_transform.update_parameters())

        # 手动更新
        for local_step_i in range(self.config["local_steps"]):
            # print("Step ", local_step_i)
            support_set_y_pred = self.rec_model(support_set_x_u, support_set_x_v,
                                              user_trans_dict=user_trans_fast_weights,
                                              item_trans_dict=item_trans_fast_weights,
                                              recomm_dict=rec_fast_weights) # 手动赋予局部更新后的参数，进行计算
            rec_loss = self.loss_function(support_set_y_pred.to(torch.float32), support_set_y.to(torch.float32))
            if for_replay:
                KD_loss = self.KD_loss_function(meta_support_logits, support_set_y_pred)
                # print(rec_loss, KD_loss)
                loss = rec_loss + self.KD_loss_ratio * KD_loss

            else:
                loss = rec_loss
                # retain为保留正向计算图，后续才可以多次调用autograd.grad()或者backforward();
            # create为计算梯度保留计算图，用于高阶求导
            grad_rec = torch.autograd.grad(loss, rec_fast_weights.values(), retain_graph = True)  # 指定参数（recomm_module）,计算梯度 (rtrain为保留计算图，为后续其他参数求导方便)(create为梯度保留计算图，为求导高阶参数)
            grad_user_trans = torch.autograd.grad(loss, user_trans_fast_weights.values(), retain_graph = True)  # 指定参数（User transformation tower）,计算梯度
            grad_item_trans = torch.autograd.grad(loss, item_trans_fast_weights.values(), retain_graph = True)  # 指定参数（Item transformation tower）,计算梯度

            rec_fast_weights = collections.OrderedDict((name,  para - self.local_lr * grads) for ((name, para), grads) in zip(rec_fast_weights.items(), grad_rec))
            user_trans_fast_weights = collections.OrderedDict((name,  para - self.local_lr * grads) for ((name, para), grads) in zip(user_trans_fast_weights.items(), grad_user_trans))
            item_trans_fast_weights = collections.OrderedDict((name,  para - self.local_lr * grads) for ((name, para), grads) in zip(item_trans_fast_weights.items(), grad_item_trans))

            # local_optimizer
            self.local_optimizer.zero_grad()  # 将之前梯度置为 0
            grad_user_emb = torch.autograd.grad(loss, self.rec_model.user_emb.parameters(), retain_graph=True)
            grad_item_emb = torch.autograd.grad(loss, self.rec_model.item_emb.parameters(), retain_graph=True)  # loss 反向传播进行梯度计算
            # loss.backward()
            self.local_optimizer.step()

            # for para_i, para_name in enumerate(rec_fast_weights.keys()):
            #     rec_fast_weights[para_name] = rec_fast_weights[para_name] - self.local_lr * grad_rec[para_i]
            # for para_i, para_name in enumerate(user_trans_fast_weights.keys()):
            #     user_trans_fast_weights[para_name] = user_trans_fast_weights[para_name] - self.local_lr * grad_user_trans[para_i]
            # for para_i, para_name in enumerate(item_trans_fast_weights.keys()):
            #     item_trans_fast_weights[para_name] = item_trans_fast_weights[para_name] - self.local_lr * grad_item_trans[para_i]


        ### Query set 计算 Meta-loss
        query_set_y_pred = self.rec_model(query_set_x_u, query_set_x_v,
                                              user_trans_dict=user_trans_fast_weights,
                                              item_trans_dict=item_trans_fast_weights,
                                              recomm_dict=rec_fast_weights)
        query_loss = self.loss_function(query_set_y_pred.to(torch.float32), query_set_y.to(torch.float32))

        if mode == "train":
            # local_optimizer 更新 & meta_optimizer 保存梯度
            self.local_optimizer.zero_grad()
            self.meta_optimizer.zero_grad()

            # print(type(self.rec_model.recomm_module.parameters()))
            # grad_rec_query = torch.autograd.grad(query_loss, self.rec_model.recomm_module.parameters(), retain_graph=True)  # 指定参数（recomm_module）,计算梯度 (rtrain为保留计算图，为后续其他参数求导方便)(create为梯度保留计算图，为求导高阶参数)
            # grad_user_trans_query  = torch.autograd.grad(query_loss, self.rec_model.user_transform.parameters(), retain_graph=True)  # 指定参数（User transformation tower）,计算梯度
            # grad_item_trans_query  = torch.autograd.grad(query_loss, self.rec_model.item_transform.parameters(), retain_graph=True)
            query_loss.backward()
            # grad_user_emb = torch.autograd.grad(local_loss, self.rec_model.user_emb.parameters(),retain_graph=True)  # loss 反向传播进行梯度计算
            # grad_item_emb = torch.autograd.grad(local_loss, self.rec_model.item_emb.parameters())
            self.local_optimizer.step()
            self.meta_optimizer.step()
            # query_grad_list = (grad_rec_query,grad_user_trans_query,grad_item_trans_query)

        query_y_real = query_set_y.detach().cpu().numpy()  # .data 是取出参数值来，.cpu() 是放到cpu上 .numpy() 是把数据从tensor转变为numpy格式
        query_y_pred = query_set_y_pred.detach().cpu().numpy()
        _mae, _rmse = self.cal_metrics.prediction(query_y_real, query_y_pred)  # sklearn.metrics
        _ndcg_3 = self.cal_metrics.ranking(query_y_real, query_y_pred, k=3)
        _ndcg_5 = self.cal_metrics.ranking(query_y_real, query_y_pred, k=5)
        _ndcg_10 = self.cal_metrics.ranking(query_y_real, query_y_pred, k=10)
        return query_loss, _mae, _rmse, _ndcg_3, _ndcg_5, _ndcg_10

    def evaluation(self, support_x_u_i, support_x_v_i, support_y_i, query_x_u_i, query_x_v_i, query_y_i,
                   device=torch.device('cpu')):
        _loss, _mae, _rmse, _ndcg_at_3, _ndcg_at_5, _ndcg_at_10 = self.local_update(support_x_u_i, support_x_v_i, support_y_i,
                                                                          query_x_u_i, query_x_v_i, query_y_i,
                                                                        mode="test", device=device)
        return _mae, _rmse, _ndcg_at_3, _ndcg_at_5, _ndcg_at_10

