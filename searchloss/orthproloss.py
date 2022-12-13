import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthogonalProjectionSearchLoss(nn.Module):
    def __init__(self, threshold):
        super(OrthogonalProjectionSearchLoss, self).__init__()
        self.threshold = threshold
    def forward(self, features, tar_probs, src_labels, search_gamma, device):
        batch_size = len(features)//2
        fs = features[:batch_size]
        ft = features[batch_size:]
        useful_ft, label_t = self.get_useful_tar(tar_probs, ft)
        if len(useful_ft)==256:
            useful_ft = useful_ft.unsqueeze(0)
            label_t = label_t.unsqueeze(0)
        f = torch.cat((fs, useful_ft), dim=0)
        l = torch.cat((src_labels, label_t))
        f = f.to(device)
        l = l.to(device)
        loss = self.loss(f, l, search_gamma, device)
        return loss

    def get_useful_tar(self, prob, feature):
        tar_softmax = nn.Softmax(dim=1)(prob)
        max_p, max_i = torch.max(tar_softmax, dim=1)
        useful_id = (max_p>self.threshold).nonzero()
        useful_id = useful_id.squeeze()
        useful_tar_f = feature[useful_id]
        useful_label = max_i[useful_id]
        return useful_tar_f, useful_label

    def loss(self, features, labels, gamma, device):
        gamma = gamma.to(device)
        features = F.normalize(features, p=2, dim=1)
        labels = labels[:, None]  # extend dim
        
        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)
        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()

        gam = gamma[labels]
        gam = gam.repeat(features.shape[0], 1)
        gam = gam.reshape(features.shape[0], features.shape[0])
        gam = (gam + gam.t())/2
        dot_prod = torch.matmul(features, features.t())

        # mask_pos = mask_pos * gam
        # pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)

        mask_neg = mask_neg * gam
        mask_pos = mask_pos * gam

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)
        loss = (1.0 - pos_pairs_mean) + neg_pairs_mean
        # if loss>1:
        #     print('mz')
        return loss



class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma
        self.threshold = 0.8

    def get_useful_tar(self, prob, feature):
        tar_softmax = nn.Softmax(dim=1)(prob)
        max_p, max_i = torch.max(tar_softmax, dim=1)
        useful_id = (max_p>self.threshold).nonzero()
        useful_id = useful_id.squeeze()
        useful_tar_f = feature[useful_id]
        useful_label = max_i[useful_id]
        return useful_tar_f, useful_label

    def forward(self, features, tar_probs, src_labels, device):
        batch_size = len(features)//2
        fs = features[:batch_size]
        ft = features[batch_size:]
        useful_ft, label_t = self.get_useful_tar(tar_probs, ft)
        if len(useful_ft)==256:
            useful_ft = useful_ft.unsqueeze(0)
            label_t = label_t.unsqueeze(0)
        features = torch.cat((fs, useful_ft), dim=0)
        labels = torch.cat((src_labels, label_t))
        features = features.to(device)
        labels = labels.to(device)

        #  features are normalized
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs

        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

        return loss

