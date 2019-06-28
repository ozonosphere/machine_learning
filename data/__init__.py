import pandas
from contextlib import contextmanager
import os
from scipy.io import loadmat

@contextmanager
def change_working_directory(path):
    old_pwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_pwd)

def get_data(file_name):
    dirname = os.path.dirname(__file__)
    full_file_path = os.path.join(dirname, file_name)
    data = pandas.read_csv(full_file_path, header=None)
    return data

def get_data_from_mat(file_name):
    dirname = os.path.dirname(__file__)
    full_file_path = os.path.join(dirname, file_name)
    data = loadmat(full_file_path)
    return data


if __name__ == '__main__':
    test = get_data_from_mat('ex3data1.mat')
    print(test)