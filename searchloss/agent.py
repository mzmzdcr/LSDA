from turtle import Turtle
import torch
import sys
sys.path.append('../..')
import common.link_utils as link
import random


class LFSAgent(object):
    def __init__(self, lr=1e-4, mean=0.0, scale=0.2, cls_num=32):

        self.counter = 0
        self.log_prob = []
        self.actions = []
        self.cls_num = cls_num

        self.mean = torch.nn.Parameter(torch.Tensor([mean,] * cls_num))  ### u
        self.delta = torch.Tensor([scale,] * cls_num)   # delta ==0.2

        self.gaussian_optimizer = torch.optim.Adam([self.mean], lr=lr, betas=(0.5, 0.999), weight_decay=0.0)
        self.relu = torch.nn.ReLU()

    def sample_subfunction(self):
        a = self.gaussian_sample_subfunction()

        return a

    def step(self, reward=0.0):
        self.gaussian_step(reward)

    def gaussian_sample_subfunction(self):
        a = []
        for i in range(self.cls_num):
            m = torch.distributions.normal.Normal(self.mean[i], self.delta[i])
            x = m.sample().item()
            x = -abs(x)
            a.append(x)
        self.actions.append(torch.tensor(a))
        return torch.tensor(a)


    def gaussian_sample_for_alpha(self):
        a_l = []
        while len(a_l)<3:
            a = self.gaussian_sample_subfunction()
            if abs(a) > abs(self.mean[0]):
                a_l.append(a)

        while len(a_l)<4:
            a = self.gaussian_sample_subfunction()
            if abs(a) < abs(self.mean[0]):
                a_l.append(a)

        i = random.choice([0, 1, 2, 3])

        return a_l[i]


    def add_multi_gaussian_log_prob(self):
        u_cuda = self.mean.cuda()
        delta_cuda = self.delta.cuda()

        actions_ = torch.stack(self.actions, dim=1).cuda()
        for i in range(self.cls_num):
            m = torch.distributions.normal.Normal(u_cuda[i], delta_cuda[i])
            self.log_prob.append(torch.sum(m.log_prob(actions_[i])))
            ## log_prob(value)是计算value在定义的正态分布（mean,1）中对应的概率的对数

    def scale_step(self, epoch, tot_epoch=10000, start_scale=0.1, final_scale=0.01):
        temp_scale = start_scale + (final_scale - start_scale) * (epoch / tot_epoch)
        self.a_delta = torch.Tensor([temp_scale, ] * 63)

    def get_lr(self, grad):
        grad_not_zero_i = grad.nonzero()
        grad_not_zero = grad[grad_not_zero_i]
        grad_ = abs(grad_not_zero)
        grad_max = grad_.max()
        lr = 1.0
        for k in range(10):
            if 1 < (grad_max * (10 ** k)) < 10 and k > 0:
                lr = 10 ** (k - 1)
                break
        return lr

    def gaussian_step(self, reward=0.0):

        self.gaussian_optimizer.zero_grad()
        self.add_multi_gaussian_log_prob()   #求每一个进程的log(g(ai,u,.))
        loss = -torch.sum(torch.stack(self.log_prob, dim=-1)) * (reward + 1e-10)
        loss.backward()
        for param in [self.mean]:  #所有进程梯度平均,上一步是loss反传获得梯度,接下来优化器更新u
            if param.requires_grad:
                param.grad.data = link.reduce_value(param.grad)
        self.gaussian_optimizer.step()
        for param in [self.mean]:
            link.broadcast(param, rank=0)
        print('self.mean:', self.mean)

        # # reset
        del self.actions[:]
        del self.log_prob[:]

    def step_v1(self, reward=0.0):
        self.add_multi_gaussian_log_prob()
        loss = -torch.sum(torch.stack(self.log_prob, dim=-1)) * (reward + 1e-10)
        loss.backward()
        for param in [self.mean]:  #所有进程梯度平均,上一步是loss反传获得梯度,接下来优化器更新u
            if param.requires_grad:
                param.grad.data = link.reduce_value(param.grad)
        self.mean.data += abs(self.mean.grad) * 0.05
        for param in [self.mean]:
            link.broadcast(param, rank=0)
        # print('self.mean:', self.mean)

        # # reset
        del self.actions[:]
        del self.log_prob[:]



class Ort_Agent(object):
    def __init__(self, lr=1e-4, mean=0.0, scale=0.2, cls_num=31):

        self.counter = 0
        self.log_prob = []
        self.actions = []
        self.cls_num = cls_num
        self.mean = torch.nn.Parameter(torch.Tensor([mean,] * cls_num))  ### u
        self.delta = torch.Tensor([scale,] * cls_num)   # delta ==0.2
        self.gaussian_optimizer = torch.optim.Adam([self.mean], lr=lr, betas=(0.5, 0.999), weight_decay=0.0)
        self.relu = torch.nn.ReLU()

    def sample_subfunction(self):
        a = self.gaussian_sample_subfunction()
        return a

    def sample_subfunction_copy(self,cls):
        a = self.gaussian_sample_subfunction_copy(cls)
        return a
    
    def gaussian_sample_subfunction_copy(self, cls):
        a = []
        for i in range(self.cls_num):
            if cls[i]:
                m = torch.distributions.normal.Normal(self.mean[i], self.delta[i])
                x = m.sample().item()
                x = -abs(x)
                a.append(x)
            else:
                a.append(self.mean[i])
        self.actions.append(torch.tensor(a))
        return torch.tensor(a)


    def step(self, reward=0.0):
        self.gaussian_step(reward)

    def gaussian_sample_subfunction(self):
        a = []
        for i in range(self.cls_num):
            m = torch.distributions.normal.Normal(self.mean[i], self.delta[i])
            x = m.sample().item()
            x = -abs(x)
            a.append(x)
        self.actions.append(torch.tensor(a))
        return torch.tensor(a)

    def add_multi_gaussian_log_prob(self):
        u_cuda = self.mean.cuda()
        delta_cuda = self.delta.cuda()

        actions_ = torch.stack(self.actions, dim=1).cuda()
        for i in range(self.cls_num):
            m = torch.distributions.normal.Normal(u_cuda[i], delta_cuda[i])
            self.log_prob.append(torch.sum(m.log_prob(actions_[i])))

    def scale_step(self, epoch, tot_epoch=10000, start_scale=0.1, final_scale=0.01):
        temp_scale = start_scale + (final_scale - start_scale) * (epoch / tot_epoch)
        self.a_delta = torch.Tensor([temp_scale, ] * 63)


    def gaussian_step(self, reward=0.0):

        self.gaussian_optimizer.zero_grad()
        self.add_multi_gaussian_log_prob()   #求每一个进程的log(g(ai,u,.))
        loss = -torch.sum(torch.stack(self.log_prob, dim=-1)) * (reward + 1e-10)
        loss.backward()
        for param in [self.mean]:  #所有进程梯度平均,上一步是loss反传获得梯度,接下来优化器更新u
            if param.requires_grad:
                param.grad.data = link.reduce_value(param.grad)
                param.grad.data = abs(param.grad.data)
        self.gaussian_optimizer.step()
        for param in [self.mean]:
            link.broadcast(param, rank=0)
        print('self.mean:', self.mean)

        # # reset
        del self.actions[:]
        del self.log_prob[:]


