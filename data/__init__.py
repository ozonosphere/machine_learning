import pandas
from contextlib import contextmanager
import os

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