from datetime import timedelta
from time import time

import utils
from training import cnn_autoencoder

# TODO create a nested dropout module
if __name__ == '__main__':
    start_time = time()
    cfg = utils.load_yaml('configs/convolutional_autoencoder_F.yaml')
    cnn_autoencoder.train_cae(cfg)
    # classifier.train_classifier()
    print(f'Total run time: {timedelta(seconds=time() - start_time)}')
