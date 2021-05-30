from datetime import timedelta
from time import time

from training import cnn_autoencoder, classifier

# TODO create a nested dropout module
if __name__ == '__main__':
    start_time = time()
    cnn_autoencoder.train_cae()
    # classifier.train_classifier()
    print(f'Total run time: {timedelta(seconds=time() - start_time)}')
