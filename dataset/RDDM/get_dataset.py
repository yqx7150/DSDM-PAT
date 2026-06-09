import argparse
from dataset.RDDM.base import Dataset
from dataset.RDDM.inpaint.dataset import InpaintDataset
from dataset.RDDM.dataset_pickel import BaseDataset as pickle_data_load
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def dataset(folder,
            image_size,
            exts=['jpg', 'jpeg', 'png', 'tiff'],
            augment_flip=False,
            convert_image_to=None,
            condition=0,
            equalizeHist=False,
            crop_patch=True,
            sample=False, 
            generation=False,
            inpaint=True):

    if inpaint:
        dataset_import = "inpaint"

    if dataset_import == "base":
        return Dataset(folder,
                       image_size,
                       exts=exts,
                       augment_flip=augment_flip,
                       convert_image_to=convert_image_to,
                       condition=condition,
                       equalizeHist=equalizeHist,
                       crop_patch=crop_patch,
                       sample=sample)

    elif dataset_import == "inpaint":
        return InpaintDataset(data_root=folder,mask_config={'mask_mode':"center"},image_size=[image_size,image_size])



def dataset_pickle(config ,condition=0,phese = 'train',data_type='AAPMDR'):

    return pickle_data_load(config , type=phese,condition=condition,split_ratio=0.99)