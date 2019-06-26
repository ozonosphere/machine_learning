import pandas

def get_data(path):
    data = pandas.read_csv(path, header=None)
    return data