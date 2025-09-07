import random
import numpy as np
import os
from tqdm import tqdm
from utils import check_and_create_path,init_seed, print_incremental_summary
import torch
import time
from DataLoader import DataLoader,TaskSampler
from MetaRecommender.MeLU.Model import Melu
import argparse
import datetime

def training(model, config, device=torch.device('cpu')):
    print('training model...')
    if config['use_cuda']:
        model.cuda()  # 模型放在GPU上
    model.train()     # 模型开启train模式
    # 加载训练数据
    ### 加载meta-training和meta-test的数据
    print('loading meta-train & test data...')
    train_data, num_train_tasks = data_loader.load_data(state='Base')  # 加载meta-training 数据
    # test_data, num_test_tasks = data_loader.load_data(state='test')  # 加载meta-test 数据
    print("Training tasks: {}".format(num_train_tasks))

    batch_size = config['batch_size']
    num_epoch = config['num_epoch']
    num_batch = int(num_train_tasks / batch_size)
    print("batch_num:{}".format(num_batch))

    for epoch_i in range(num_epoch):
        print('start epoch {}:'.format(epoch_i), datetime.datetime.now())
        loss, mae, rmse, ndcg_at_3, ndcg_at_5, ndcg_at_10 = [],[],[],[],[],[]
        # shuffle train data
        random.shuffle(train_data)

        for batch_i in tqdm(range(num_batch)):
            train_tasks_bi = train_data[batch_size*batch_i: batch_size*(batch_i + 1)]
            _loss, mae_i, rmse_i, ndcg_3_i, ndcg_5_i, ndcg_10_i = model.global_update(train_tasks_bi, device=device)  # 每个batch上进行一次全局更新

            # 记录每个 iteration 的 metrics
            loss.append(_loss)
            mae.append(mae_i)
            rmse.append(rmse_i)
            ndcg_at_3.append(ndcg_3_i)
            ndcg_at_5.append(ndcg_5_i)
            ndcg_at_10.append(ndcg_10_i)

        # training tasks的query set上的结果
        model.scheduler.step()
        print('epoch: {}, loss: {:.6f}, mae: {:.5f}, rmse: {:.5f},  ndcg_3: {:.5f}, ndcg_5: {:.5f}, ndcg_10: {:.5f}'.format(epoch_i, np.mean(loss), np.mean(mae), np.mean(rmse), np.mean(ndcg_at_3), np.mean(ndcg_at_5), np.mean(ndcg_at_10)))  # report整个epoch的结果，为meat-training数据上的损失和评价参数

        # if epoch_i % 1 == 0:
        #     print('evaluating model...')
        #     _pre_test_mae, _post_test_mae, _pre_test_ndcg_3, _post_test_ndcg_3 = testing(model, test_data, device)
        #     print('Epoch {}: pre_test_mae: {:.5f}, post_test_mae: {:.5f}, pre_test_ndcg: {:.5f}, post_test_ndcg: {:.5f}'.format(epoch_i, _pre_test_mae, _post_test_mae, _pre_test_ndcg_3, _post_test_ndcg_3))
        #     model.train()     # 切换为train模式

    print("*******************start saving model!************************")
    saved_model_filename = "saved/{}/MeLU_Base_trail1.pkl".format(config["dataset"])  # 保存模型参数
    check_and_create_path(saved_model_filename)
    saved_model = model.state_dict()
    print(saved_model.keys())
    torch.save(saved_model, saved_model_filename)
    print("*******************Model saved in {}!************************".format(saved_model_filename))

def Incremental_update(model, span, train_data, train_task_num, device=torch.device('cpu')):
    if config['use_cuda']:
        model.cuda()
    model.eval()   # eval模式

    batch_size = config['batch_size']
    num_epoch = config['incremental_update_epoch']
    num_batch = int(train_task_num / batch_size)

    for epoch_i in range(num_epoch):
        print('start epoch {}:'.format(epoch_i), datetime.datetime.now())
        loss, mae, rmse, ndcg_at_3, ndcg_at_5, ndcg_at_10 = [], [], [], [], [], []
        random.shuffle(train_data)

        for batch_i in tqdm(range(num_batch)):
            train_tasks_bi = train_data[batch_size*batch_i: batch_size*(batch_i + 1)]
            _loss, mae_i, rmse_i, ndcg_3_i, ndcg_5_i, ndcg_10_i = model.global_update(train_tasks_bi, device=device)  # 每个batch上进行一次全局更新

            # 记录每个 iteration 的 metrics
            loss.append(_loss)
            mae.append(mae_i)
            rmse.append(rmse_i)
            ndcg_at_3.append(ndcg_3_i)
            ndcg_at_5.append(ndcg_5_i)
            ndcg_at_10.append(ndcg_10_i)

        # model.scheduler.step()
        print('Train epoch: {}, loss: {:.6f}, mae: {:.5f}, rmse: {:.5f},  ndcg_3: {:.5f}, ndcg_5: {:.5f}, ndcg_10: {:.5f}'.format(
                epoch_i, np.mean(loss), np.mean(mae), np.mean(rmse), np.mean(ndcg_at_3), np.mean(ndcg_at_5),
                np.mean(ndcg_at_10)))  # report整个epoch的结果，为meat-training数据上的损失和评价参数

    # print("****************** Saving model!************************")
    saved_model_filename = "saved/{}/Replay_Fine_tune/{}/MeLU_Incrental_{}.pkl".format(config["dataset"],'trail1',span)  # 保存模型参数
    check_and_create_path(saved_model_filename)
    saved_model = model.state_dict()
    # print(saved_model.keys())
    torch.save(saved_model, saved_model_filename)
    print("*******************Model saved in {}!************************".format(saved_model_filename))


def test(model, test_data, device=torch.device('cpu')):
    if config['use_cuda']:
        model.cuda()
    model.eval()   # eval模式

    task_ids, support_xs_u, support_xs_v, support_ys, query_xs_u, query_xs_v, query_ys = zip(*test_data)

    mae_s, rmse_s, ndcg_at_3_s, ndcg_at_5_s, ndcg_at_10_s = [], [], [], [], []
    for i in range(len(task_ids)):  # each task
        _mae, _rmse, _ndcg_at_3, _ndcg_at_5, _ndcg_at_10 = model.evaluation(support_xs_u[i], support_xs_v[i],
                                                                            support_ys[i], query_xs_u[i],
                                                                            query_xs_v[i],
                                                                            query_ys[i],
                                                                            device=device)  # 对每个任务进行local update, 并返回query loss
        mae_s.append(_mae)
        rmse_s.append(_rmse)
        ndcg_at_3_s.append(_ndcg_at_3)
        ndcg_at_5_s.append(_ndcg_at_5)
        ndcg_at_10_s.append(_ndcg_at_10)

    # Test Metrics Display
    # print('mae: {:.5f}, rmse: {:.5f}, ndcg@1: {:.5f}, ndcg@3: {:.5f}, ndcg@5: {:.5f}'.
    #       format(np.mean(mae), np.mean(rmse), np.mean(ndcg_at_1), np.mean(ndcg_at_3), np.mean(ndcg_at_5)))
    return mae_s, rmse_s, ndcg_at_3_s, ndcg_at_5_s, ndcg_at_10_s

def Incremetal_train(model, config, device=torch.device('cpu')):
    print('Incremetal_train model...')
    if config['use_cuda']:
        model.cuda()  # 模型放在GPU上

    result_summary = {}
    time_span_list = range(config['start_span'], config['end_span'] + 1)


    reservior_sampler = TaskSampler(config)
    old_data, old_task_num = data_loader.load_data(state="Base")
    elapsed_time_list = []

    ### 先测试，后增量更新
    for span_i in time_span_list:
        start_time_i = time.time()  # 记录开始时间
        # new tasks
        new_data, new_num_tasks = data_loader.load_data(state="Incremental_{}".format(span_i))  # 加载meta-test 数据
        print("Incremental_Span: {}, New tasks: {}".format(span_i, new_num_tasks))
        ### 测试
        mae_s, rmse_s, ndcg_at_3_s, ndcg_at_5_s, ndcg_at_10_s = test(model, new_data, device=device)
        result_summary[span_i] = (np.mean(mae_s), np.mean(rmse_s), np.mean(ndcg_at_3_s), np.mean(ndcg_at_5_s), np.mean(ndcg_at_10_s))
        print('Test_Metrics: {}, mae: {:.5f}, rmse: {:.5f},  ndcg_3: {:.5f}, ndcg_5: {:.5f}, ndcg_10: {:.5f}'.format(
                span_i, np.mean(mae_s), np.mean(rmse_s), np.mean(ndcg_at_3_s), np.mean(ndcg_at_5_s),
                    np.mean(ndcg_at_10_s)))  # report整个epoch的结果，为meat-training数据上的损失和评价参数
        
        # replay taks
        print("train_tasks", len(old_data))
        replay_data = reservior_sampler.random_sample(old_data, config["memory_size"])

        ### 增量更新
        data = new_data + replay_data
        num_tasks = len(new_data + replay_data)
        print("train_tasks", num_tasks)
        Incremental_update(model, span_i, data, num_tasks, device=device)
        old_data = data

        end_time_i = time.time()  # 记录结束时间
        elapsed_time_i = end_time_i - start_time_i  # 计算程序运行时间
        elapsed_time_list.append(elapsed_time_i)
        print("增量时段: {} 增量训练时间为：{}".format(span_i, elapsed_time_i), "秒")

    print_incremental_summary(result_summary)
    print(elapsed_time_list)

def Incremetal_test(model, config, device=torch.device('cpu')):
    if config['use_cuda']:
        model.cuda()
    model.eval()   # eval模式

    # train_data, num_train_tasks = data_loader.load_data(state='Base')  # 加载meta-training 数据
    time_span_list = range(config['start_span'],config['end_span']+1)
    for span_i in time_span_list:
        test_data, num_test_tasks = data_loader.load_data(state="Incremental_{}".format(span_i))  # 加载meta-test 数据
        print("Span: {}, Test tasks: {}".format(span_i, num_test_tasks))

        task_ids, support_xs_u, support_xs_v, support_ys, query_xs_u, query_xs_v, query_ys = zip(*test_data)

        mae_s, rmse_s, ndcg_at_3_s, ndcg_at_5_s, ndcg_at_10_s = [], [], [], [], []
        for i in range(len(task_ids)):  # each task
            _mae, _rmse, _ndcg_at_3, _ndcg_at_5, _ndcg_at_10 = model.evaluation(support_xs_u[i], support_xs_v[i],
                                                                                        support_ys[i], query_xs_u[i],
                                                                                         query_xs_v[i],
                                                                                         query_ys[i], device=device)  # 对每个任务进行local update, 并返回query loss
            mae_s.append(_mae)
            rmse_s.append(_rmse)
            ndcg_at_3_s.append(_ndcg_at_3)
            ndcg_at_5_s.append(_ndcg_at_5)
            ndcg_at_10_s.append(_ndcg_at_10)

        print('Incremental_span_: {}, mae: {:.5f}, rmse: {:.5f},  ndcg_3: {:.5f}, ndcg_5: {:.5f}, ndcg_10: {:.5f}'.format(span_i, np.mean(mae_s), np.mean(rmse_s), np.mean(ndcg_at_3_s), np.mean(ndcg_at_5_s), np.mean(ndcg_at_10_s)))  # report整个epoch的结果，为meat-training数据上的损失和评价参数


if __name__ == "__main__":
    init_seed(2023)  # 固定随机种子
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("Start time:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Movielens10M', help='Movielens10M/')
    parser.add_argument('--Incremental_learner', default='Fine_tune', help='')
    parser.add_argument('--model_name', default='MeLU', help='')
    opt = parser.parse_args()

    # 加载Config
    if opt.dataset == "Movielens10M":
        from Configs.Configurations_Movielens10M_Fine_Tune import Movielens10M_config as config
    # elif opt.dataset == "dbook":
    #     from Configs.Configurations_Dbook import Dbook_config as config
    # elif opt.dataset == "yelp":
    #     from  Configs.Configurations_Yelp import Yelp_config as config
    print("Configurations:")
    print(config)

    # 加载数据
    Input_dir = config["Input_dir"]
    data_loader = DataLoader(Input_dir,config)  # Original DataLoader

    ### 加载模型
    print('--------------- Load Model {} ---------------'.format(opt.model_name))
    stage = "Pretrain"
    model = Melu(config, opt.model_name, stage)

    # ### 测试
    # print('--------------- Incremetal Test ---------------')
    # cuda_or_cpu = torch.device("cuda" if config['use_cuda'] else "cpu")
    # Incremetal_test(model, config, device=cuda_or_cpu)

    ### 增量更新
    print('--------------- Incremetal Update ---------------')
    cuda_or_cpu = torch.device("cuda" if config['use_cuda'] else "cpu")
    Incremetal_train(model, config, device=cuda_or_cpu)