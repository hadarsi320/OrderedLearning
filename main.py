from datetime import timedelta
from time import time

from clearml import Task

import utils
from training import cnn_autoencoder

# TODO create a nested dropout module
if __name__ == '__main__':
    start_time = time()
    original_cfg = utils.load_yaml('configs/conv_ae_vanilla.yaml')

    for mode in ['F', 'G', 'H']:
        cfg = original_cfg.copy()
        cfg['model']['mode'] = mode

        task = Task.init(project_name='Ordered Learning')
        task.connect_configuration(cfg, name='Model Configuration')
        cnn_autoencoder.train_cae(cfg, task)
        task.close()
        print(f'Total run time: {timedelta(seconds=time() - start_time)}')
