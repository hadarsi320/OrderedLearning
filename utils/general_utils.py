import time
from datetime import datetime

import torch
import yaml


def binarize_data(data: torch.Tensor, bin_quantile=0.5):
    if not 0 < bin_quantile < 1:
        raise ValueError('The binarization quantile must be in range (0, 1)')

    binarized = torch.zeros_like(data)
    q_quantiles = torch.quantile(data, bin_quantile, dim=0)
    binarized[data >= q_quantiles] = 1
    return binarized


def get_current_time():
    return datetime.now().strftime('%y-%m-%d--%H-%M-%S')


def get_device():
    if torch.cuda.is_available():
        if torch.cuda.device_count() == 8:
            return torch.device('cuda:7')
        return torch.device('cuda')
    return torch.device('cpu')


def format_number(number, precision=3):
    if number == 0:
        return number

    round_number = round(float(number), ndigits=precision)
    if round_number != 0:
        return round_number

    return f'{number:.{precision}e}'


def format_time(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))


def load_yaml(path) -> dict:
    return yaml.load(open(path), yaml.Loader)


def dump_yaml(data, path):
    yaml.dump(data, open(path, 'w'), sort_keys=False)
