import pandas as pd
import random
from datetime import datetime
import os
import pickle

def check_and_create_path(filename):
    file_dir = os.path.split(filename)[0]
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)

'''
    基于recbole提供的Atomfile, 继续进行数据预处理和切分
'''
def timespan_selection(raw_file,saved_file):
    # convert csv into pandas dataframe & transfer timestamp to date
    df = pd.read_csv(raw_file, sep='\t')
    df['date'] = df['timestamp:float'].apply(lambda x: int(datetime.fromtimestamp(x).strftime('%Y%m%d')))
    df.columns = ['userId', 'itemId', 'rating', 'timestamp', 'date']  # rename
    print(df.head(20))

    # 查看总天数 并 筛选数据
    date_list = sorted(df['date'].unique().tolist()) #
    print(len(date_list))
    print(max(date_list),min(date_list))

    start_date = 20060101  # 三年
    end_date = 20090101
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    print(df.shape)

    # 按月划分数据
    user_ids = sorted(df['userId'].unique().tolist())
    item_ids = sorted(df['itemId'].unique().tolist())
    print(len(user_ids))  # 14886
    print(len(item_ids))  # 10529

    # 切分时段，交互打标签 timespan
    start_datetime = datetime.strptime("2006-01-01","%Y-%m-%d")
    df['timespan'] = df['timestamp'].apply(lambda x: int((datetime.fromtimestamp(x).year-start_datetime.year)*12 + (datetime.fromtimestamp(x).month-start_datetime.month)))
    print(df['timespan'].unique().tolist())

    '''
       数据暂存点-1: 时段数据切分
    '''
    check_and_create_path(saved_file)
    df.to_csv(saved_file,index = False)

def cold_start_selection(raw_file, saved_file, low_threshold, high_threshold):
    df = pd.read_csv(raw_file)
    print(len(df['userId'].unique().tolist()))  # 15000
    print(len(df['itemId'].unique().tolist()))  # 10500

    # 1. 每个用户 保留 第一次timespan的交互
    def drop_late_timespans(x):
        x= x.sort_values(by='timestamp', ascending=True)
        times_spans = sorted(x['timespan'].unique())[0]
        # return x.drop(x[x.times_spans.isin(times_spans)].index)
        return x[x["timespan"] == times_spans]
    df = df.groupby("userId").apply(drop_late_timespans).reset_index(drop=True) # reset_index(drop=True) 将旧索引扔掉，替换成整数索引
    print(df.shape)

    # 2. 筛选 冷启动用户 (筛掉zero-shot users)
    user_counts = df["userId"].value_counts()
    zero_coldstart_uders = user_counts[user_counts < low_threshold].index
    print("zero_coldstart_users", len(zero_coldstart_uders))
    few_coldstart_users = user_counts[(user_counts>=low_threshold) & (user_counts <= high_threshold)].index
    print("coldstart_users",len(few_coldstart_users))
    warm_users = user_counts[user_counts > high_threshold].index
    print("warm_users", len(warm_users))

    users_to_keep = user_counts[user_counts>=low_threshold].index
    df = df[df.userId.isin(users_to_keep)]

    # 3. warm users 截断多余交互
    def truncate_warm(x):
        return x.iloc[:high_threshold]
    df = df.groupby("userId").apply(truncate_warm).reset_index(drop=True)
    print(df.shape)

    '''
        数据暂存点-2: 冷启动数据筛选
    '''
    check_and_create_path(saved_file)
    df.to_csv(saved_file,index = False)

def incremental_process(input_file, saved_base_file, saved_incremental_file, split_ratio, remap_file_user, remap_file_item):
    df = pd.read_csv(input_file)

    user_ids = sorted(df['userId'].unique().tolist())
    item_ids = sorted(df['itemId'].unique().tolist())
    print(len(user_ids))  # 12375
    print(len(item_ids))  # 7521
    print(df.shape)  # 534362

    # 按timespan统计用户和数据
    span_values = df["timespan"].value_counts()
    user_counts = df.groupby("timespan").apply(lambda x:len(x["userId"].unique()))
    print(user_counts)

    # remap users
    user_ids = sorted(df['userId'].unique().tolist())  # 排序 users
    print(max(user_ids))
    user_remap = dict(zip(user_ids, range(len(user_ids))))  # create map, key is original id, value is mapped id starting from 0
    df['userId'] = df['userId'].map(lambda x: user_remap[x])  # map key to value in df
    print(max(df['userId']))

    # remap items
    item_ids = sorted(df['itemId'].unique().tolist())  # 排序 users
    print(max(item_ids))
    item_remap = dict(zip(item_ids, range(len(item_ids))))  # create map, key is original id, value is mapped id starting from 0
    df['itemId'] = df['itemId'].map(lambda x: item_remap[x])  # map key to value in df
    print(max(df['itemId']))

    '''
        数据存储点 3: remap 文件
    '''
    check_and_create_path(remap_file_user)
    pickle.dump(user_remap, open(remap_file_user, "wb"))
    pickle.dump(item_remap, open(remap_file_item, "wb"))

    # 按时段 切分数据集 [Base_data, incremental data]
    span_ids = sorted(df['timespan'].unique().tolist())
    base_df = df[df.timespan.isin(span_ids[:int(len(span_ids)*split_ratio)])]
    incre_df = df[df.timespan.isin(span_ids[int(len(span_ids)*split_ratio):])]
    print(base_df.shape) # 281563
    print(incre_df.shape) # 252799

    '''
        数据暂存点-4: Base-data & Incremental data
    '''
    check_and_create_path(saved_base_file)
    base_df.to_csv(saved_base_file,index = False)
    incre_df.to_csv(saved_incremental_file,index = False)

def Item_features(attribute_file, remap_dict, saved_file):
    attribute_df = pd.read_csv(attribute_file, sep='\t')
    attribute_df.columns= ['item', 'title', 'year', 'genre']
    print(attribute_df)

    year_list = attribute_df["year"].unique().tolist()
    print(len(year_list)) # 94
    genres = attribute_df["genre"].unique().tolist()
    genre_list = ["(no genres listed)"]
    for _genra in genres:
        if str(_genra) != "(no genres listed)":
            sub_strs = str(_genra).split(" ")
            for _str in sub_strs:
                if _str not in genre_list:
                    genre_list.append(_str)
    print(len(genre_list))
    print(genre_list)

    item_remap = pickle.load(open(remap_dict, "rb"))
    print(item_remap)

    item_fea = {}
    for _item, item_df in attribute_df.iterrows():
        if item_df['item'] in item_remap:
            # print(item_df)
            origi_id = item_df['item']
            remap_id = item_remap[origi_id]

            item_year = year_list.index(item_df["year"])
            item_genres = [0] * len(genre_list)
            if str(item_df["genre"]) == "(no genres listed)":
                item_genres[0] = 1
            else:
                for _genre in str(item_df["genre"]).split(" "):
                    idx = genre_list.index(_genre)
                    item_genres[idx] = 1
            item_fea[remap_id] = [remap_id,item_year] + item_genres
            # print(item_fea)
    print(len(item_fea))
    '''
        数据暂存点-5: Item_features
    '''
    check_and_create_path(saved_file)
    pickle.dump(item_fea, open(saved_file, "wb"))

def generate_cold_start_meta_tasks(data, taskfile, feature_file, support_len):
    item_features = pickle.load(open(feature_file, "rb"))
    grouped_users = data.groupby("userId")

    all_tasks = []
    for user_id, interactions in grouped_users:
        # Support set
        support_interactions = interactions.iloc[:support_len]
        support_ratings = support_interactions["rating"].tolist()
        support_user_features = support_interactions["userId"].tolist()
        support_item_features = []
        for item_i in support_interactions["itemId"].tolist():
            support_item_features.append(item_features[item_i])

        # Query set
        query_interactions = interactions.iloc[support_len:]
        query_ratings = query_interactions["rating"].tolist()
        query_user_features =  query_interactions["userId"].tolist()
        query_item_features = []
        for item_i in  query_interactions["itemId"].tolist():
            query_item_features.append(item_features[item_i])

        all_tasks.append((user_id, support_user_features,support_item_features, support_ratings,query_user_features, query_item_features, query_ratings)) # (Task_ID, support_x_u, support_x_v, support_y, query_x_u, query_x_v, query_y)
    print(len(all_tasks))

    '''
        数据暂存点-6: Cold-start Meta Tasks
    '''
    check_and_create_path(taskfile)
    pickle.dump(all_tasks,open(taskfile, "wb"))

if __name__ == '__main__':
    random.seed(2023)

    '''
        常规数据处理
    '''
    ### Step 1: 时段数据划分
    # interaction_file = '../Data/raw/Movielens10M/ml-10m.inter'
    # saved_incremental_datafile = '../Data/processed/Movielens10M/ml-10m_incre_inter.csv'
    # timespan_selection(interaction_file,saved_incremental_datafile)

    ### Step 2: 冷启动数据筛选
    # incremental_datafile = '../Data/processed/Movielens10M/ml-10m_incre_inter.csv'
    # coldstart_datafile = '../Data/processed/Movielens10M/ml-10m_coldstart_inter.csv'
    # cold_start_threshold = [20, 50]
    # cold_start_selection(incremental_datafile, coldstart_datafile, cold_start_threshold[0], cold_start_threshold[1])

    ### Step 3: 按时段整理数据 至 【Base data, incremental data】，并Remap users & items
    # coldstart_datafile = '../Data/processed/Movielens10M/ml-10m_coldstart_inter.csv'
    # user_remap_file = '../Data/processed/Movielens10M/user_remap.csv'
    # item_remap_file = '../Data/processed/Movielens10M/item_remap.csv'
    # Base_datafile = '../Data/processed/Movielens10M/ml-10m_Base_data.csv'
    # Incremental_datafile = '../Data/processed/Movielens10M/ml-10m_Incremental_data.csv'
    # Base_Incre_ratio = 0.5
    # incremental_process(coldstart_datafile, Base_datafile, Incremental_datafile, Base_Incre_ratio, user_remap_file,item_remap_file)

    ### Step 4: Item Features (Movielens只有item attributes)
    # item_attribute_file = '../Data/raw/Movielens10M/ml-10m.item'
    # item_remap_file = '../Data/processed/Movielens10M/item_remap.csv'
    # saved_feature_file = '../Data/processed/Movielens10M/Item_features.pkl'
    # Item_features(item_attribute_file, item_remap_file, saved_feature_file)

    '''
        元学习模型 数据处理
    '''
    ### Step 5: 将Base data & incremental data 处理成 Base tasks & Incremental_tasks
    Base_datafile = '../Data/processed/Movielens10M/ml-10m_Base_data.csv'
    item_feature_file = '../Data/processed/Movielens10M/Item_features.pkl'
    Base_taskfile = '../Data/processed/Movielens10M/Base_tasks.pkl'
    Base_data = pd.read_csv(Base_datafile)
    support_len = 10
    generate_cold_start_meta_tasks(Base_data, Base_taskfile, item_feature_file, support_len)

    ### Step 6: 每个incremental_span 分解一个incremental_tasks
    Incremental_datafile = '../Data/processed/Movielens10M/ml-10m_Incremental_data.csv'
    item_feature_file = '../Data/processed/Movielens10M/Item_features.pkl'
    Incremental_taskfile = '../Data/processed/Movielens10M/Incremental_{}_tasks.pkl'
    support_len = 10
    Incremental_data = pd.read_csv(Incremental_datafile)
    span_ids = Incremental_data["timespan"].unique().tolist()
    for span_i in span_ids:
        print(span_i)
        generate_cold_start_meta_tasks(Incremental_data[Incremental_data["timespan"] == span_i], Incremental_taskfile.format(span_i), item_feature_file, support_len)