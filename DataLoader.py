import pickle
import random

import numpy as np

class DataLoader:
    def __init__(self, input_dir, config):
        self.input_dir = input_dir
        self.config = config

    def load_data(self, state):
        print("loading data")
        # 正常加载test tasks
        total_data = pickle.load(open(self.input_dir + "/{}_tasks.pkl".format(state), "rb"))

        # 转换为array （使用时再转会比较慢）
        task_id_all, support_x_u_all, support_x_v_all, support_y_all, query_x_u_all, query_x_v_all, query_y_all = zip(*total_data)
        task_id_list = []
        support_x_u_tensor = []
        support_x_v_tensor = []
        support_y_tensor = []
        query_x_u_tensor = []
        query_x_v_tensor = []
        query_y_tensor = []

        num_tasks = len(task_id_all)
        query_length = 0

        for i in range(num_tasks):
            task_id_list.append( np.array(task_id_all[i]))
            support_x_u_tensor.append(np.array(support_x_u_all[i]))
            support_x_v_tensor.append(np.array(support_x_v_all[i]))
            support_y_tensor.append(np.array(support_y_all[i]))

            query_x_u_tensor.append(np.array(query_x_u_all[i]))
            query_x_v_tensor.append(np.array(query_x_v_all[i]))
            query_y_tensor.append(np.array(query_y_all[i]))
            query_length += len(query_y_all[i])

        print("avg. query len", query_length / num_tasks)
        total_data_tensor = list(zip(task_id_list, support_x_u_tensor, support_x_v_tensor, support_y_tensor, query_x_u_tensor, query_x_v_tensor, query_y_tensor))
        return total_data_tensor, num_tasks

class TaskSampler:
    def __init__(self, config):
        self.config = config
        self.reservior = []
        self.reservior_size = config["memory_size"]

    def random_sample(self, old_data, sample_num):
        sampled_tasks = random.sample(old_data, sample_num)
        return sampled_tasks

