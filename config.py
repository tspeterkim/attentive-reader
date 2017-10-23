import argparse

def str2bool(v):
    return v.lower() in ['yes','true','1','t','y']

def get_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool) # TODO: what does this exactly do

    return parser.parse_args()
