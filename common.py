import os
from pathlib import Path
import time
from torch.utils.data import random_split
from functools import wraps

def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        t_beg = time.time()
        result = f(*args, **kwargs)
        t_end = time.time()
        print(f'{f.__name__} t = {t_end - t_beg}')
        return result
    return wrapper

class Timer:
    def __init__(self, message=''):
        self.start_time = time.time()
        self.last_time = self.start_time
    
    def get(self, message='', update_time=True):
        t = time.time()
        print(f'{message}, dt = {t - self.last_time:4.3f}, all_time = {t - self.start_time: 5.3}', flush=True)
        if update_time:
            self.last_time = t


def change_cwd(file):
    if file is None:
        return
    else:
        os.chdir(os.path.abspath(os.path.dirname(file)))
    print('current working directory: ', os.path.abspath(os.getcwd()))

def my_random_split(dataset, lengths):
    if len(lengths) == 2 and list_type_is(lengths, [int, float]):
        coef = len(dataset)/sum(lengths)
        new_lengths = [int(coef * v) for v in lengths]
        new_lengths[0] = len(dataset) - sum(new_lengths[1:])
        return random_split(dataset, new_lengths)
    elif len(lengths) == 3 and list_type_is(lengths, [int, float]):
        coef = len(dataset)/sum(lengths)
        new_lengths = [int(coef * v) for v in lengths]
        new_lengths[0] = len(dataset) - sum(new_lengths[1:])
        train, not_train = random_split(dataset, (new_lengths[0], sum(new_lengths[1:])))
        test, valid = my_random_split(not_train, new_lengths[1:])
        return train, test, valid

    return None

def list_type_is(list_, type_list):
    if type(list_) not in [list, tuple]:
        type_list = [type_list]
    for v in list_:
        if type(v) not in type_list:
            return False
    return True

