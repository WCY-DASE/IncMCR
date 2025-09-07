Movielens10M_config = {
    'dataset': 'Movielens10M',
    "Input_dir" : "../../Data/Final/Movielens10M/",
    # "Test_input_dir": "../../dataset/movielens/meta_cold_start_10",
    # "test_model_file": "../../saved/movielens/baselines/MeLU/MeLU_movielens_Aug.pkl",
    'use_cuda': True,
    # 'model_save': False,

    # Incremental_Setup
    'start_span': 18,
    'end_span': 35,

    ### feature dims
    # item
    'num_item': 7521,
    'num_year': 94,
    'num_genre': 20,

    # user
    'num_user': 12375,

    ### model settings
    # embedding
    'base_model': 'Two_tower',
    'embedding_dim': 32,
    'user_embedding_dim': 32,  # 1 features
    'item_embedding_dim': 32*3,  # 3 features

    # recomm module
    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,

    # Meta_learning Setting
    'Meta_learned_para': ["recomm", "user_transform","item_transform"],

    # Training settings
    'local_steps': 1,
    'global_lr': 5e-4,
    'local_lr': 1e-3,
    'lr_ml_step':[10,15], #
    'lr_dc': 0.1, # 以0.1 进行lr衰减

    'batch_size': 16,  # for each batch, the number of tasks
    'num_epoch': 10,
    'l2': 1e-5,
}