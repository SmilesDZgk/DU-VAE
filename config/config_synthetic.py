
params={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 2,
    'ni': 50,
    'enc_nh': 50,
    'dec_nh': 50,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    'epochs': 50,
    'batch_size': 32,
    'test_nepoch': 1,
    'train_data': 'data/synthetic_data/synthetic_train.txt',
    'val_data': 'data/synthetic_data/synthetic_test.txt',
    'test_data': 'data/synthetic_data/synthetic_test.txt',
    'vocab_file': 'data/synthetic_data/vocab.txt',
    "label": True
}
