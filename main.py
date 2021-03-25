from datetime import timedelta
from time import time

from training import fc_autoencoder

if __name__ == '__main__':
    start_time = time()
    # run whatever
    fc_autoencoder.main()
    print(f'Total run time: {timedelta(seconds=time() - start_time)}')