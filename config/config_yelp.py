
params={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 512,
    'enc_nh': 1024,
    'dec_nh': 1024,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    'batch_size': 32,
    'epochs': 150,
    'test_nepoch': 5,
    'train_data': 'data/yelp_data/yelp.train.txt',
    'val_data': 'data/yelp_data/yelp.valid.txt',
    'test_data': 'data/yelp_data/yelp.test.txt',
    'vocab_file': 'data/yelp_data/vocab.txt',
    'label':True
}


params_ss_10={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 512,
    'enc_nh': 1024,
    'dec_nh': 1024,
    'log_niter': 50,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    # 'batch_size': 32,
    #'epochs': 100,
    'test_nepoch': 5,
    'train_data': 'data/yelp_data/yelp.train.10.txt',
    'val_data': 'data/yelp_data/yelp.valid.txt',
    'test_data': 'data/yelp_data/yelp.test.txt',
    'vocab_file': 'data/yelp_data/vocab.txt',
    'ncluster': 5,
    "label": True
}


params_ss_100={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 512,
    'enc_nh': 1024,
    'dec_nh': 1024,
    'log_niter': 50,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    # 'batch_size': 32,
    #'epochs': 100,
    'test_nepoch': 5,
    'train_data': 'data/yelp_data/yelp.train.100.txt',
    'val_data': 'data/yelp_data/yelp.valid.txt',
    'test_data': 'data/yelp_data/yelp.test.txt',
    'vocab_file': 'data/yelp_data/vocab.txt',
    'ncluster': 5,
    "label": True
}

params_ss_500={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 512,
    'enc_nh': 1024,
    'dec_nh': 1024,
    'log_niter': 50,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    #'batch_size': 50,
    #'epochs': 100,
    'test_nepoch': 5,
    'train_data': 'data/yelp_data/yelp.train.500.txt',
    'val_data': 'data/yelp_data/yelp.valid.txt',
    'test_data': 'data/yelp_data/yelp.test.txt',
    'vocab_file': 'data/yelp_data/vocab.txt',
    'ncluster': 5,
    "label": True
}

params_ss_1000={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 512,
    'enc_nh': 1024,
    'dec_nh': 1024,
    'log_niter': 50,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    # 'batch_size': 32,
    #'epochs': 100,
    'test_nepoch': 5,
    'train_data': 'data/yelp_data/yelp.train.1000.txt',
    'val_data': 'data/yelp_data/yelp.valid.txt',
    'test_data': 'data/yelp_data/yelp.test.txt',
    'vocab_file': 'data/yelp_data/vocab.txt',
    'ncluster': 5,
    "label": True
}


params_ss_2000={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 512,
    'enc_nh': 1024,
    'dec_nh': 1024,
    'log_niter': 50,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    # 'batch_size': 32,
    #'epochs': 100,
    'test_nepoch': 5,
    'train_data': 'data/yelp_data/yelp.train.2000.txt',
    'val_data': 'data/yelp_data/yelp.valid.txt',
    'test_data': 'data/yelp_data/yelp.test.txt',
    'vocab_file': 'data/yelp_data/vocab.txt',
    'ncluster': 5,
    "label": True
}


params_ss_10000={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 512,
    'enc_nh': 1024,
    'dec_nh': 1024,
    'log_niter': 50,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    # 'batch_size': 32,
    #'epochs': 100,
    'test_nepoch': 5,
    'train_data': 'data/yelp_data/yelp.train.10000.txt',
    'val_data': 'data/yelp_data/yelp.valid.txt',
    'test_data': 'data/yelp_data/yelp.test.txt',
    'vocab_file': 'data/yelp_data/vocab.txt',
    'ncluster': 5,
    "label": True
}