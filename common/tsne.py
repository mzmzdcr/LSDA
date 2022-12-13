import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm
import matplotlib
import os

matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = '0'



def visualize(feature: torch.Tensor, filename: str, source_color='r', target_color='b'):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    feature = feature.cpu().numpy()
    y = np.arange(31)
    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(feature)

    # visualize using matplotlib
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, s=2)
    plt.savefig(filename)

label = torch.zeros(93,31)
for i in range(31):
    a = 3 * i
    b = a + 3
    label[a:b][i]=i
# plot t-SNE
save_path = '/remote-home/meizhen/10.26.2.244/'
tSNE_filename = os.path.join(save_path, 'label.png')
visualize(label, tSNE_filename)
print("Saving t-SNE to", tSNE_filename)