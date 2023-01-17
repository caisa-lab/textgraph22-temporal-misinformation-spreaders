import argparse

def compute_idx_from_time(date):
    add = 0
    if date.year == 2021:
        add = 12

    idx = date.month - 1 + add
    return idx

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def convert_value(value):
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, list):
        return [convert_value(val) for val in value]
    else:
        return str(value)

#TODO Maybe write to json? The issue is at the dataset_descriptor
def convert_dict_to_json(dictionary):
    raise NotImplemented("Method not implemented")
    pass