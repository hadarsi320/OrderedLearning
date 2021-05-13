from datetime import timedelta
from time import time

from training import cnn_autoencoder, classifier

if __name__ == '__main__':
    start_time = time()
    # cnn_autoencoder.main()
    classifier.main()
    print(f'Total run time: {timedelta(seconds=time() - start_time)}')
