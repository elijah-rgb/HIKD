import torch
import torch.nn as nn
import torchvision
from scipy.spatial import distance_matrix
import numpy as np
from tqdm import tqdm
import os
import sys
import torch
import numpy as np
import tensorflow as tf
from models.models_dict import DATASET_MODELS_DICT
from models.model_helpers import get_model, get_optimizer, get_domain_extractors

"""
This code allows you to train single domain learning networks.
"""

import os
import sys
import torch
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data.meta_dataset_reader import (MetaDatasetBatchReader,
                                      MetaDatasetEpisodeReader)
from models.losses import cross_entropy_loss, prototype_loss
from models.model_utils import (CheckPointer, UniformStepLR,
                                CosineAnnealRestartLR, ExpDecayLR)
from models.model_helpers import get_model, get_optimizer
from utils import Accumulator
from config import args


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def delta_hyp(dismat):
    """
    computes delta hyperbolicity value from distance matrix
    """

    p = 0
    row = dismat[p, :][np.newaxis, :]
    col = dismat[:, p][:, np.newaxis]
    XY_p = 0.5 * (row + col - dismat)

    maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)
    return np.max(maxmin - XY_p)


def batched_delta_hyp(X, n_tries=10, batch_size=1500):
    vals = []
    for i in tqdm(range(n_tries)):
        idx = np.random.choice(len(X), batch_size)
        X_batch = X[idx]
        distmat = distance_matrix(X_batch, X_batch)
        diam = np.max(distmat)
        delta_rel = 2 * delta_hyp(distmat) / diam
        vals.append(delta_rel)
    return np.mean(vals), np.std(vals)


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B = x.shape[0]
        return x.view(B, -1)


def get_delta(loader):
    """
    computes delta value for image data by extracting features using VGG network;
    input -- data loader for images
    """
    vgg = torchvision.models.vgg16(pretrained=True)
    vgg_feats = vgg.features
    vgg_classifier = nn.Sequential(*list(vgg.classifier.children())[:-1])

    vgg_part = nn.Sequential(vgg_feats, Flatten(), vgg_classifier).to(device)
    vgg_part.eval()

    all_features = []
    for i, (batch) in enumerate(loader):
        with torch.no_grad():
            batch = batch.to(device)
            all_features.append(vgg_part(batch).detach().cpu().numpy())

    all_features = np.concatenate(all_features)
    idx = np.random.choice(len(all_features), 1500)
    all_features_small = all_features[idx]

    dists = distance_matrix(all_features_small, all_features_small)
    delta = delta_hyp(dists)
    diam = np.max(dists)
    return delta, diam


"""
This code allows you to train single domain learning networks.
"""




def train():

    tf.compat.v1.disable_eager_execution()
    # Setting up datasets
    trainsets, valsets, testsets = ['ilsvrc_2012'],['ilsvrc_2012'],['ilsvrc_2012']

    trainsets = ['ilsvrc_2012']
    test_loader = MetaDatasetEpisodeReader('test', trainsets, trainsets, testsets, test_type=args['test.type'])
    model = get_model(None, args)
    checkpointer = CheckPointer(args, model, optimizer=None)
    checkpointer.restore_model(ckpt='best', strict=False)
    model.eval()
    model.cuda()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = False
    with tf.compat.v1.Session(config=config) as session:
 
        sample = test_loader.get_test_task(session, trainsets)

        all_features = []
        for i, (batch) in enumerate(sample['context_images']):
            with torch.no_grad():
                batch = batch.to(device)
                all_features.append(model.embed(sample['context_images']).detach().cpu().numpy())

        all_features = np.concatenate(all_features)
        idx = np.random.choice(len(all_features), 500)
        all_features_small = all_features[idx]

        dists = distance_matrix(all_features_small, all_features_small)
        delta = delta_hyp(dists)
        diam = np.max(dists)
    return delta, diam
    
if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()

    train()
