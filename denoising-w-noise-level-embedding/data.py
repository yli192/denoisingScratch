import os
from dataset import DataLoaderTrain3d_wNoiseInfo_l1, DataLoaderVal3d_wNoiseInfo_l1


def get_training_data3d_wNoiseInfo_l1(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain3d_wNoiseInfo_l1(rgb_dir, img_options)

def get_validation_data3d_wNoiseInfo_l1(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal3d_wNoiseInfo_l1(rgb_dir, img_options)

