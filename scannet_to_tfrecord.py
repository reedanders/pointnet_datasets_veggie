import numpy as np
import tensorflow

import train
from utils import provider

def generate_train_data():

    # Shuffle train samples
    dataset = train.TRAIN_DATASET
    import pdb; pdb.set_trace()
    
    for ii in len(dataset):
    
        ps,seg,smpw = dataset[ii]
        aug_data = provider.rotate_point_cloud_z(ps)


def generate_test_data():
    pass


def generate_whole_test_data():
    pass


if __name__ == "__main__":

    generate_train_data()    
