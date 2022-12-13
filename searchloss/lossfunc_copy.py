import sys
sys.path.append('../')
from .agent import LFSAgent, Ort_Agent
from .loss import *
from .orthproloss import OrthogonalProjectionSearchLoss
#
# from agent import LFSAgent
# from loss import search_loss
# from orthproloss import OrthogonalProjectionSearchLoss
import torch
import math
import common.link_utils as link
import torch.distributed as dist


class LossFuncSearch(object):
    def __init__(self, sm=1, threshold=0.8, a=-10., t=0.35, cls_num=31):
        self.model = None
        self.lr = 0.05
        self.sample_step = 2
        self.val_freq = 2
        self.scale = 0.2
        self.global_rank = dist.get_rank()
        self.best_acc = 0
        self.best_epoch = -1
        self.sm = sm
        self.cls_num = cls_num
        self.__init_agent()
        self.__init_oploss(threshold)
        self.gamma = torch.tensor([1.0, ] * cls_num)
        self.a = a
        self.t = t
        self.feat_alpha = torch.tensor([0.])
        self.pred_alpha = torch.tensor([0.])
        

    def __init_agent(self):
        self.gamma_agent = Ort_Agent(self.lr, -1.0, self.scale, self.cls_num)

        self.feat_alpha_agent = LFSAgent(self.lr, -0.0, self.scale, 1)
        self.pred_alpha_agent = LFSAgent(self.lr, -0.0, self.scale, 1)
    
    def __init_oploss(self, threshold):
        self.oploss = OrthogonalProjectionSearchLoss(threshold)

    def oploss_forward(self, features, tar_probs, src_labels, device):

        loss = self.oploss(features, tar_probs, src_labels, self.gamma, device)
        return loss

    def set_model(self, model):
        self.model = model

    def change_alpha(self, feat=True):
        if feat:
            self.feat_alpha = torch.tensor([self.a])
        else:
            self.pred_alpha = torch.tensor([self.a])

    def adv_loss_pred(self, x, y):
        loss = binary_adv_loss(x, y, self.pred_alpha, self.t, False)
        return loss

    def adv_loss_feat(self, feat, softout, lbl, adnet, device, randomlayer, entropy, coeff):
        loss = feature_adloss([feat, softout], lbl, adnet, device, randomlayer, entropy, coeff, self.feat_alpha, t=0.35)
        return loss

    def dann_feat_loss(self, feat, lbl, adnet):
        loss = dann_feat_loss(feat, lbl, adnet, self.feat_alpha, t=0.35)
        return loss

    def set_alpha_parameters(self, epoch, device=None):
        if epoch > 1:
            self.feat_alpha = self.feat_alpha_agent.sample_subfunction()
            self.feat_alpha = (self.feat_alpha * math.exp(self.sm)).to(device)
            self.pred_alpha = self.pred_alpha_agent.sample_subfunction()
            self.feat_alpha = (self.feat_alpha * math.exp(self.sm)).to(device)
            print('self.feat_alpha', self.feat_alpha)
            
    def set_gamma_parameters(self, epoch, device=None):
        if epoch > 1:
            self.gamma = self.gamma_agent.sample_subfunction()
            self.gamma = (-self.gamma * math.exp(1)).to(device)
        print('self.gamma', self.gamma)
            

    def _broadcast_parameters(self, rank):
        link.broadcast_params(self.model, rank)

    def update_lfs(self, reward):
        rank = self.global_rank
        temp_acc = torch.tensor(reward).cuda(rank)

        test_acc_tensor = link.all_gather(temp_acc)
        print('div', test_acc_tensor)
        # best_test_acc_rank = torch.argmax(test_acc_tensor)
        best_test_acc_rank = torch.argmin(test_acc_tensor)
        print('sync rank:', best_test_acc_rank)
        self._broadcast_parameters(rank=best_test_acc_rank.item())
        reward = (test_acc_tensor - torch.mean(test_acc_tensor)) / ((torch.max(test_acc_tensor) - torch.min(test_acc_tensor)) + 1e-6) * 2

        print('reward:', -reward)

        self.feat_alpha_agent.step(reward=-reward[rank].item())
        self.pred_alpha_agent.step(reward=-reward[rank].item())
        self.gamma_agent.step(reward=-reward[rank].item())


    def update_lfs_max(self, reward):
        rank = self.global_rank
        temp_acc = torch.tensor(reward).cuda(rank)

        test_acc_tensor = link.all_gather(temp_acc)
        print('div', test_acc_tensor)
        best_test_acc_rank = torch.argmax(test_acc_tensor)
        # best_test_acc_rank = torch.argmin(test_acc_tensor)
        print('sync rank:', best_test_acc_rank)
        self._broadcast_parameters(rank=best_test_acc_rank.item())
        reward = (test_acc_tensor - torch.mean(test_acc_tensor)) / ((torch.max(test_acc_tensor) - torch.min(test_acc_tensor)) + 1e-6) * 2

        print('reward:', reward)

        self.feat_alpha_agent.step(reward=reward[rank].item())
        self.pred_alpha_agent.step(reward=reward[rank].item())
        self.gamma_agent.step(reward=reward[rank].item())

    def update_lfs_ablation(self, reward):
        rank = self.global_rank
        temp_acc = torch.tensor(reward).cuda(rank)

        test_acc_tensor = link.all_gather(temp_acc)
        print('div', test_acc_tensor)
        best_test_acc_rank = torch.argmax(test_acc_tensor)
        # best_test_acc_rank = torch.argmin(test_acc_tensor)
        print('sync rank:', best_test_acc_rank)
        self._broadcast_parameters(rank=best_test_acc_rank.item())
        reward = (test_acc_tensor - torch.mean(test_acc_tensor)) / ((torch.max(test_acc_tensor) - torch.min(test_acc_tensor)) + 1e-6) * 2

        print('reward:', reward)

        # self.feat_alpha_agent.step(reward=reward[rank].item())
        # self.pred_alpha_agent.step(reward=reward[rank].item())
        self.gamma_agent.step(reward=reward[rank].item())

    def get_best_acc_rank(self, reward):
        rank = self.global_rank
        temp_acc = torch.tensor(reward).cuda(rank)

        test_acc_tensor = link.all_gather(temp_acc)
        best_test_acc_rank = torch.argmax(test_acc_tensor)

        return test_acc_tensor[best_test_acc_rank]
