import numpy as np
import argparse
import sys
def load_argparser_args(conf_dirs):
    parser = argparse.ArgumentParser()
    for conf_dir in conf_dirs:
        module = __import__(conf_dir, fromlist=['init_arguments'])
        f = module.__dict__['init_arguments']
        f(parser)

    flags = parser.parse_args()
    return flags
