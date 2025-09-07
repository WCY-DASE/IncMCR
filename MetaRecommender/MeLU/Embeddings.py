import torch
from torch.autograd import Variable

'''
Dbook Embeddings
'''
class UserEmbeddingDB(torch.nn.Module): # 同样继承torch.nn.Module作为embedding layer的模块
    def __init__(self, config):
        super(UserEmbeddingDB, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.num_user = config['num_user']
        self.num_location = config['num_location']
        self.embedding_userId = torch.nn.Embedding(
            num_embeddings=self.num_user,
            embedding_dim=self.embedding_dim
        )
        self.embedding_location = torch.nn.Embedding(
            num_embeddings=self.num_location,
            embedding_dim=self.embedding_dim
        )

    def forward(self, user_fea):
        """
        """
        userId_idx = Variable(user_fea[:, 0], requires_grad=False)
        location_idx = Variable(user_fea[:, 1], requires_grad=False)  # 将输入的特征分离，形成不同的Variable，指定为不计算梯度

        userId_emb = self.embedding_userId(userId_idx)  # 各自通过embedding layer
        location_emb = self.embedding_location(location_idx)
        return torch.cat((userId_emb, location_emb), 1)  # (samples, 2*32)  torch.cat() concatenation

    def Get_ID_emb(self, user_ID):
        userId_idx = Variable(user_ID, requires_grad=False)
        userId_emb = self.embedding_userId(userId_idx)
        return userId_emb

    def Get_Attri_emb(self, user_fea):
        location_idx = Variable(user_fea[:, 1], requires_grad=False)  # 将输入的特征分离，形成不同的Variable，指定为不计算梯度
        location_emb = self.embedding_location(location_idx)
        return location_emb # (samples, 2*32)  torch.cat() concatenation

class UserEmbeddingDB_ID(torch.nn.Module):  # 同样继承torch.nn.Module作为embedding layer的模块
    def __init__(self, config):
        super(UserEmbeddingDB_ID, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.num_user = config['num_user']
        self.embedding_userId = torch.nn.Embedding(
            num_embeddings=self.num_user,
            embedding_dim=self.embedding_dim
        )

    def forward(self, user_fea):
        """
        """
        userId_idx = Variable(user_fea[:, 0], requires_grad=False)
        userId_emb = self.embedding_userId(userId_idx)
        return userId_emb

class UserEmbeddingDB_Attribute(torch.nn.Module):  # 同样继承torch.nn.Module作为embedding layer的模块
    def __init__(self, config):
        super(UserEmbeddingDB_Attribute, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.num_location = config['num_location']
        self.embedding_location = torch.nn.Embedding(
            num_embeddings=self.num_location,
            embedding_dim=self.embedding_dim
        )

    def forward(self, user_fea):
        """
            传入的数据是一样的，会有ID和attribute，只是embedding时，利用的数据不同
        """
        location_idx = Variable(user_fea[:, 1], requires_grad=False)  # 将输入的特征分离，形成不同的Variable，指定为不计算梯度
        location_emb = self.embedding_location(location_idx)
        return location_emb

class ItemEmbeddingDB(torch.nn.Module):
    def __init__(self, config):
        super(ItemEmbeddingDB, self).__init__()
        self.num_item = config['num_item']
        self.num_author = config['num_author']
        self.num_publisher = config['num_publisher']
        self.num_year = config['num_year']

        self.embedding_dim = config['embedding_dim']

        self.embedding_itemId = torch.nn.Embedding(
            num_embeddings=self.num_item,
            embedding_dim=self.embedding_dim
        )
        self.embedding_author = torch.nn.Embedding(
            num_embeddings=self.num_author,
            embedding_dim=self.embedding_dim
        )
        self.embedding_publisher = torch.nn.Embedding(
            num_embeddings=self.num_publisher,
            embedding_dim=self.embedding_dim
        )
        self.embedding_year = torch.nn.Embedding(
            num_embeddings=self.num_year,
            embedding_dim=self.embedding_dim
        )

    def forward(self, item_fea):
        """
        :param item_fea: [ID(0),rate（1），genre（2，2+num_genre），actor(2+num_genre,2+num_genre+num_actors), director(2+num_genre+num_actors: ) ]
        :return:
        """

        itemId_idx = Variable(item_fea[:, 0], requires_grad=False)
        author_idx = Variable(item_fea[:, 1], requires_grad=False)
        publisher_idx = Variable(item_fea[:, 2], requires_grad=False)
        year_idx = Variable(item_fea[:, 3], requires_grad=False)

        itemId_emb = self.embedding_itemId(itemId_idx) # (1,32)
        author_emb = self.embedding_author(author_idx)  # (1,32)
        publisher_emb = self.embedding_publisher(publisher_idx)  # (1,32)
        year_emb = self.embedding_year(year_idx)  # (1,32)

        return torch.cat((itemId_emb,author_emb, publisher_emb,year_emb), 1)  # (samples, 5*32)

class ItemEmbeddingDB_ID(torch.nn.Module):
    def __init__(self, config):
        super(ItemEmbeddingDB_ID, self).__init__()
        self.num_item = config['num_item']
        self.embedding_dim = config['embedding_dim']
        self.embedding_itemId = torch.nn.Embedding(
            num_embeddings=self.num_item,
            embedding_dim=self.embedding_dim
        )

    def forward(self, item_fea):
        """
        :param item_fea: [ID(0),rate（1），genre（2，2+num_genre），actor(2+num_genre,2+num_genre+num_actors), director(2+num_genre+num_actors: ) ]
        :return:
        """

        itemId_idx = Variable(item_fea[:, 0], requires_grad=False)
        itemId_emb = self.embedding_itemId(itemId_idx) # (1,32)
        return itemId_emb  # (samples, 5*32)

'''
     Yelp Embeddings
'''
class UserEmbeddingYP(torch.nn.Module): # 同样继承torch.nn.Module作为embedding layer的模块
    def __init__(self, config):

        super(UserEmbeddingYP, self).__init__()
        # 每种feature的类别数量
        self.num_user = config['num_user']
        self.num_fans = config['num_fans']
        self.num_avgrating = config['num_avgrating']

        self.embedding_dim = config['embedding_dim']

        self.embedding_userId = torch.nn.Embedding(
            num_embeddings=self.num_user,
            embedding_dim=self.embedding_dim
        )
        self.embedding_fans = torch.nn.Embedding(
            num_embeddings=self.num_fans,
            embedding_dim=self.embedding_dim
        )
        self.embedding_avgrating = torch.nn.Embedding(
            num_embeddings=self.num_avgrating,
            embedding_dim=self.embedding_dim
        )

    def forward(self, user_fea):
        """
        :param user_fea: [userId(0),gender(1),age(2),occupation(3),area(4)]
        :return:
        """
        userId_idx = Variable(user_fea[:, 0], requires_grad=False)
        fans_idx = Variable(user_fea[:, 1], requires_grad=False) # 将输入的特征分离，形成不同的Variable，指定为不计算梯度
        avgrating_idx = Variable(user_fea[:, 2], requires_grad=False) # 将输入的特征分离，形成不同的Variable，指定为不计算梯度

        userId_emb = self.embedding_userId(userId_idx)  # 各自通过embedding layer
        fans_emb = self.embedding_fans(fans_idx)
        avgrating_emb = self.embedding_avgrating(avgrating_idx)
        return torch.cat((userId_emb,fans_emb,avgrating_emb), 1)   # (samples, 3*32)  torch.cat() concatenation

    def Get_ID_emb(self, user_ID):
        userId_idx = Variable(user_ID, requires_grad=False)
        userId_emb = self.embedding_userId(userId_idx)  # 各自通过embedding layer
        return userId_emb  # (samples, 3*32)  torch.cat() concatenation

    def Get_Attri_emb(self, user_fea):
        fans_idx = Variable(user_fea[:, 1], requires_grad=False)  # 将输入的特征分离，形成不同的Variable，指定为不计算梯度
        avgrating_idx = Variable(user_fea[:, 2], requires_grad=False)  # 将输入的特征分离，形成不同的Variable，指定为不计算梯度

        fans_emb = self.embedding_fans(fans_idx)
        avgrating_emb = self.embedding_avgrating(avgrating_idx)
        return torch.cat((fans_emb, avgrating_emb), 1)  # (samples, 3*32)  torch.cat() concatenation

class UserEmbeddingYP_Attribute(torch.nn.Module): # 同样继承torch.nn.Module作为embedding layer的模块
    def __init__(self, config):
        super(UserEmbeddingYP_Attribute, self).__init__()
        # 每种feature的类别数量
        self.num_fans = config['num_fans']
        self.num_avgrating = config['num_avgrating']

        self.embedding_dim = config['embedding_dim']

        self.embedding_fans = torch.nn.Embedding(
            num_embeddings=self.num_fans,
            embedding_dim=self.embedding_dim
        )
        self.embedding_avgrating = torch.nn.Embedding(
            num_embeddings=self.num_avgrating,
            embedding_dim=self.embedding_dim
        )

    def forward(self, user_fea):
        """
        :param user_fea: [userId(0),gender(1),age(2),occupation(3),area(4)]
        :return:
        """
        fans_idx = Variable(user_fea[:, 1], requires_grad=False) # 将输入的特征分离，形成不同的Variable，指定为不计算梯度
        avgrating_idx = Variable(user_fea[:, 2], requires_grad=False) # 将输入的特征分离，形成不同的Variable，指定为不计算梯度

        fans_emb = self.embedding_fans(fans_idx)
        avgrating_emb = self.embedding_avgrating(avgrating_idx)
        return torch.cat((fans_emb,avgrating_emb), 1)   # (samples, 3*32)  torch.cat() concatenation

class UserEmbeddingYP_ID(torch.nn.Module): # 同样继承torch.nn.Module作为embedding layer的模块
    def __init__(self, config):
        super(UserEmbeddingYP_ID, self).__init__()
        # 每种feature的类别数量
        self.num_user = config['num_user']
        self.embedding_dim = config['embedding_dim']

        self.embedding_userId = torch.nn.Embedding(
            num_embeddings=self.num_user,
            embedding_dim=self.embedding_dim
        )

    def forward(self, user_fea):
        """
        :param user_fea: [userId(0),gender(1),age(2),occupation(3),area(4)]
        :return:
        """
        userId_idx = Variable(user_fea[:, 0], requires_grad=False)
        userId_emb = self.embedding_userId(userId_idx) #各自通过embedding layer
        return userId_emb # (samples, 3*32)  torch.cat() concatenation

class ItemEmbeddingYP(torch.nn.Module):
    def __init__(self, config):
        super(ItemEmbeddingYP, self).__init__()
        self.num_item = config['num_item']
        self.num_postalcode = config['num_postalcode']
        self.num_stars = config['num_stars']
        self.num_city = config['num_city']
        self.num_category = config['num_category']

        self.embedding_dim = config['embedding_dim']

        self.embedding_itemId = torch.nn.Embedding(
            num_embeddings=self.num_item,
            embedding_dim=self.embedding_dim
        )
        self.embedding_postalcode = torch.nn.Embedding(
            num_embeddings=self.num_postalcode,
            embedding_dim=self.embedding_dim
        )
        self.embedding_stars = torch.nn.Embedding(
            num_embeddings=self.num_stars,
            embedding_dim=self.embedding_dim
        )
        self.embedding_city = torch.nn.Embedding(
            num_embeddings=self.num_city,
            embedding_dim=self.embedding_dim
        )
        self.embedding_category = torch.nn.Linear(
            in_features=self.num_category,
            out_features=self.embedding_dim,
            bias=False
        )

    def forward(self, item_fea):
        """
        :param item_fea: [ID(0),rate（1），genre（2，2+num_genre），actor(2+num_genre,2+num_genre+num_actors), director(2+num_genre+num_actors: ) ]
        :return:
        """

        itemId_idx = Variable(item_fea[:, 0], requires_grad=False)
        postalcode_idx = Variable(item_fea[:, 1], requires_grad=False)
        stars_idx = Variable(item_fea[:, 2], requires_grad=False)
        city_idx = Variable(item_fea[:, 3], requires_grad=False)
        category_idx = Variable(item_fea[:, 4:], requires_grad=False)

        itemId_emb = self.embedding_itemId(itemId_idx) # (1,32)
        postalcode_emb = self.embedding_postalcode(postalcode_idx)  # (1,32)
        stars_emb = self.embedding_stars(stars_idx)  # (1,32)
        city_emb = self.embedding_city(city_idx)  # (1,32)

        category_sum = torch.sum(category_idx.float(), 1).view(-1, 1)
        category_emb = self.embedding_category(category_idx.float()) / torch.where(category_sum == 0, torch.ones_like(category_sum),
                                                                          category_sum)  # (1,32) view() 相当于 reshape

        return torch.cat((itemId_emb,postalcode_emb, stars_emb,city_emb,category_emb), 1)  # (samples, 5*32)

class ItemEmbeddingYP_ID(torch.nn.Module):
    def __init__(self, config):
        super(ItemEmbeddingYP_ID, self).__init__()
        self.num_item = config['num_item']
        self.embedding_dim = config['embedding_dim']

        self.embedding_itemId = torch.nn.Embedding(
            num_embeddings=self.num_item,
            embedding_dim=self.embedding_dim
        )
    def forward(self, item_fea):
        """
        :param item_fea: [ID(0),rate（1），genre（2，2+num_genre），actor(2+num_genre,2+num_genre+num_actors), director(2+num_genre+num_actors: ) ]
        :return:
        """

        itemId_idx = Variable(item_fea[:, 0], requires_grad=False)
        itemId_emb = self.embedding_itemId(itemId_idx) # (1,32)

        return itemId_emb

'''
    Movielens Embeddings
'''
# class UserEmbeddingML(torch.nn.Module): # 同样继承torch.nn.Module作为embedding layer的模块
#     def __init__(self, config):
#         super(UserEmbeddingML, self).__init__()
#         # 每种feature的类别数量
#         self.num_user = config['num_user']
#         self.num_gender = config['num_gender']
#         self.num_age = config['num_age']
#         self.num_occupation = config['num_occupation']
#         self.num_zipcode = config['num_zipcode']
#         self.embedding_dim = config['embedding_dim']
#
#         self.embedding_userId = torch.nn.Embedding(
#             num_embeddings=self.num_user,
#             embedding_dim=self.embedding_dim
#         )  # torch.nn.Embedding() 构建embedding_layer
#         self.embedding_gender = torch.nn.Embedding(
#             num_embeddings=self.num_gender,
#             embedding_dim=self.embedding_dim
#         ) # torch.nn.Embedding() 构建embedding_layer
#         self.embedding_age = torch.nn.Embedding(
#             num_embeddings=self.num_age,
#             embedding_dim=self.embedding_dim
#         )
#         self.embedding_occupation = torch.nn.Embedding(
#             num_embeddings=self.num_occupation,
#             embedding_dim=self.embedding_dim
#         )
#         self.embedding_area = torch.nn.Embedding(
#             num_embeddings=self.num_zipcode,
#             embedding_dim=self.embedding_dim
#         )
#
#     def forward(self, user_fea):
#         """
#         :param user_fea: [userId(0),gender(1),age(2),occupation(3),area(4)]
#         :return:
#         """
#         userId_idx = Variable(user_fea[:, 0], requires_grad=False)
#         gender_idx = Variable(user_fea[:, 1], requires_grad=False) # 将输入的特征分离，形成不同的Variable，指定为不计算梯度
#         age_idx = Variable(user_fea[:, 2], requires_grad=False)
#         occupation_idx = Variable(user_fea[:, 3], requires_grad=False)
#         area_idx = Variable(user_fea[:, 4], requires_grad=False)
#
#         userId_emb = self.embedding_userId(userId_idx)  # 各自通过embedding layer
#         gender_emb = self.embedding_gender(gender_idx)
#         age_emb = self.embedding_age(age_idx)
#         occupation_emb = self.embedding_occupation(occupation_idx)
#         area_emb = self.embedding_area(area_idx)
#         return torch.cat((userId_emb,gender_emb, age_emb, occupation_emb, area_emb), 1)   # (samples, 5*32)  torch.cat() concatenation
#
#     def Get_ID_emb(self, user_ID):
#         userId_idx = Variable(user_ID, requires_grad=False)
#         userId_emb = self.embedding_userId(userId_idx)
#         return userId_emb  # (samples, 32)
#
#     def Get_Attri_emb(self, user_fea):
#         gender_idx = Variable(user_fea[:, 1], requires_grad=False)  # 将输入的特征分离，形成不同的Variable，指定为不计算梯度
#         age_idx = Variable(user_fea[:, 2], requires_grad=False)
#         occupation_idx = Variable(user_fea[:, 3], requires_grad=False)
#         area_idx = Variable(user_fea[:, 4], requires_grad=False)
#
#         gender_emb = self.embedding_gender(gender_idx)
#         age_emb = self.embedding_age(age_idx)
#         occupation_emb = self.embedding_occupation(occupation_idx)
#         area_emb = self.embedding_area(area_idx)
#         return torch.cat((gender_emb, age_emb, occupation_emb, area_emb),1)  # (samples, 5*32)  torch.cat() concatenation

class UserEmbeddingML(torch.nn.Module): # 同样继承torch.nn.Module作为embedding layer的模块
    def __init__(self, config):
        super(UserEmbeddingML, self).__init__()
        # 每种feature的类别数量
        self.num_user = config['num_user']
        self.embedding_dim = config['embedding_dim']

        vars = torch.nn.ParameterDict()
        self.embedding_userId = torch.nn.Embedding(
            num_embeddings=self.num_user,
            embedding_dim=self.embedding_dim
        )  # torch.nn.Embedding() 构建embedding_layer
        vars['user_embedding'] = self.embedding_userId.weight
        self.vars = vars

    def forward(self, user_fea):
        """
        :param user_fea: [userId(0)]
        :return:
        """
        userId_idx = Variable(user_fea, requires_grad=False)
        userId_emb = self.embedding_userId(userId_idx)
        return userId_emb   # (samples, emb_dim)

    def update_parameters(self):  # 获取可更新参数（local update）
        return self.vars

class UserEmbeddingML_Attribute(torch.nn.Module): # 同样继承torch.nn.Module作为embedding layer的模块
    def __init__(self, config):
        super(UserEmbeddingML_Attribute, self).__init__()
        # 每种feature的类别数量
        self.num_gender = config['num_gender']
        self.num_age = config['num_age']
        self.num_occupation = config['num_occupation']
        self.num_zipcode = config['num_zipcode']
        self.embedding_dim = config['embedding_dim']
        self.embedding_gender = torch.nn.Embedding(
            num_embeddings=self.num_gender,
            embedding_dim=self.embedding_dim
        ) # torch.nn.Embedding() 构建embedding_layer
        self.embedding_age = torch.nn.Embedding(
            num_embeddings=self.num_age,
            embedding_dim=self.embedding_dim
        )
        self.embedding_occupation = torch.nn.Embedding(
            num_embeddings=self.num_occupation,
            embedding_dim=self.embedding_dim
        )
        self.embedding_area = torch.nn.Embedding(
            num_embeddings=self.num_zipcode,
            embedding_dim=self.embedding_dim
        )

    def forward(self, user_fea):
        """
        :param user_fea: [userId(0),gender(1),age(2),occupation(3),area(4)]
        :return:
        """
        gender_idx = Variable(user_fea[:, 1], requires_grad=False) # 将输入的特征分离，形成不同的Variable，指定为不计算梯度
        age_idx = Variable(user_fea[:, 2], requires_grad=False)
        occupation_idx = Variable(user_fea[:, 3], requires_grad=False)
        area_idx = Variable(user_fea[:, 4], requires_grad=False)

        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        area_emb = self.embedding_area(area_idx)
        return torch.cat((gender_emb, age_emb, occupation_emb, area_emb), 1)   # (samples, 5*32)  torch.cat() concatenation

class ItemEmbeddingML(torch.nn.Module):
    def __init__(self, config):
        super(ItemEmbeddingML, self).__init__()
        self.num_item = config['num_item']
        self.num_year = config['num_year']
        self.num_genre = config['num_genre']

        self.embedding_dim = config['embedding_dim']

        vars = torch.nn.ParameterDict()
        self.embedding_itemId = torch.nn.Embedding(
            num_embeddings=self.num_item,
            embedding_dim=self.embedding_dim
        )
        vars['item_embedding'] = self.embedding_itemId.weight

        self.embedding_year = torch.nn.Embedding(
            num_embeddings=self.num_year,
            embedding_dim=self.embedding_dim
        )
        vars['year_embedding'] = self.embedding_year.weight

        self.embedding_genre = torch.nn.Linear(
            in_features=self.num_genre,
            out_features=self.embedding_dim,
            bias=False
        ) # torch.nn.Linear() 对张量的最后一维进行线性变化，multi-hot做embedding的技巧
        vars['genre_embedding'] = self.embedding_genre.weight

        self.vars = vars

    def forward(self, item_fea):
        """
        :param item_fea: [ID(0), year（1），genre（2，2+num_genre) ]
        :return:
        """
        itemId_idx = Variable(item_fea[:, 0], requires_grad=False)
        year_idx = Variable(item_fea[:, 1], requires_grad=False)
        genre_idx = Variable(item_fea[:, 2:2+self.num_genre], requires_grad=False) # 离散的multi_hot

        itemId_emb = self.embedding_itemId(itemId_idx) # (1,32)
        year_emb = self.embedding_year(year_idx)  # (1,32)
        genre_sum = torch.sum(genre_idx.float(), 1).view(-1, 1)
        genre_emb = self.embedding_genre(genre_idx.float()) / torch.where(genre_sum==0, torch.ones_like(genre_sum) , genre_sum)  # (1,32) view() 相当于 reshape
        return torch.cat((itemId_emb,year_emb, genre_emb), 1)  # (samples, 3*emb_dim)

    def update_parameters(self):  # 获取可更新参数（local update）
        return self.vars

class ItemEmbeddingML_ID(torch.nn.Module):
    def __init__(self, config):
        super(ItemEmbeddingML_ID, self).__init__()
        self.num_item = config['num_item']
        self.embedding_dim = config['embedding_dim']

        self.embedding_itemId = torch.nn.Embedding(
            num_embeddings=self.num_item,
            embedding_dim=self.embedding_dim
        )

    def forward(self, item_fea):
        """
        :param item_fea: [ID(0),rate（1），genre（2，2+num_genre），actor(2+num_genre,2+num_genre+num_actors), director(2+num_genre+num_actors: ) ]
        :return:
        """

        itemId_idx = Variable(item_fea[:, 0], requires_grad=False)
        itemId_emb = self.embedding_itemId(itemId_idx) # (1,32)
        return itemId_emb # (samples, 5*32)
