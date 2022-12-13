
import math
import warnings
warnings.filterwarnings("ignore")
import sys
import os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append(os.path.dirname(parentdir))
import random
import datetime
import pytz
import warnings
import sys
import argparse
import numpy as np
import copy
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader, sampler
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
import os.path as osp
from searchloss.mmd import MMD_loss
from searchloss.lossfunc_copy import LossFuncSearch
from searchloss.loss import feat_adout, binary_entropy
import Network.netzoo as NetZoo
from network import ImageClassifier
import backbone as BackboneNetwork
from utils import ContinuousDataloader
from transforms import ResizeImage
from lr_scheduler import LrScheduler
from data_list import ImageList, make_weights_for_balanced_classes, VISDA_ImageList
import warnings
from common.ema import EMA
warnings.filterwarnings("ignore")

def get_current_time():
    return datetime.datetime.now(pytz.timezone('PRC')).strftime("%Y-%m-%d_%H:%M")

def set_seed(num):  
    random.seed(num)         
    torch.manual_seed(num)         
    cudnn.deterministic = True     
    cudnn.benchmark = True


def main(args, config, rank):
    global save_dir
    save_dir = args.output_dir
    
    backbone = BackboneNetwork.__dict__[args.arch](pretrained=True)
    classifier = ImageClassifier(backbone, args.num_classes)  
    hidden_num = classifier.features_dim
    feat_adnet = NetZoo.AdversarialNetwork(hidden_num, 256)
    pred_adnet = NetZoo.PAdversarialNetwork(args.num_classes, 256)
    model = [classifier, feat_adnet, pred_adnet]

    all_parameters = classifier.get_parameters() + feat_adnet.get_parameters() + pred_adnet.get_parameters()
    optimizer = SGD(all_parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_sheduler = LrScheduler(optimizer, init_lr=args.lr, gamma=0.001, decay_rate=0.75)
    
    lfs = LossFuncSearch(args.sm, 0.8, t=args.t, cls_num=args.num_classes)
    lfs.set_model(model)
    best_acc = 0

    # load data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.center_crop:
        train_transform = transforms.Compose([ResizeImage(256), 
                                            transforms.CenterCrop(224), 
                                            transforms.RandomHorizontalFlip(), 
                                            transforms.ToTensor(), 
                                            normalize])
    else:
        train_transform = transforms.Compose([ResizeImage(256), 
                                            transforms.RandomResizedCrop(224), 
                                            transforms.RandomHorizontalFlip(), 
                                            transforms.ToTensor(), 
                                            normalize])

    val_tranform = transforms.Compose([ResizeImage(256), 
                                    transforms.CenterCrop(224), 
                                    transforms.ToTensor(), 
                                    normalize])

    train_source_dataset = ImageList(open(args.s_dset_path).readlines(), transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_dataset = ImageList(open(args.t_dset_path).readlines(), transform=train_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)

    val_dataset = ImageList(open(args.val_path).readlines(), transform=val_tranform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    if args.dset == 'domainnet':
        test_dataset = ImageList(open(args.t_dset_path).readlines(), transform=val_tranform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=64)
    else:
        test_loader = val_loader

    train_source_iter = ContinuousDataloader(train_source_loader)
    train_target_iter = ContinuousDataloader(train_target_loader)

    device = torch.device("cuda:%d" % rank)
    classifier = nn.DataParallel(classifier, device_ids=[rank]).to(device)
    feat_adnet = nn.DataParallel(feat_adnet, device_ids=[rank]).to(device)
    pred_adnet = nn.DataParallel(pred_adnet, device_ids=[rank]).to(device)
    # mmd_ema = EMA(0.95, 0)

    for epoch in range(args.epochs):
        if epoch> 10:
            lfs.set_gamma_parameters(epoch, device)

        train(train_source_iter, train_target_iter, classifier, optimizer, feat_adnet, pred_adnet, lfs, lr_sheduler, epoch, args, device)
        curr_acc = validate(test_loader, classifier, device)
        if epoch > 10:
            domain_discrenpancy = discriminator_capacity(train_loader, classifier, feat_adnet, device)
            lfs.update_lfs_ablation(domain_discrenpancy)

        print("curr_acc={:.3f}, best_acc = {:.3f}".format(curr_acc, best_acc))
        config["out_file"].write("curr_acc={:.3f}, best_acc = {:.3f}".format(curr_acc, best_acc) + '\n')
        config["out_file"].flush()

    torch.cuda.empty_cache()
    print("best_acc = {:.3f}".format(best_acc))
    config["out_file"].write("best_acc = {:.3f}".format(best_acc) + '\n')
    config["out_file"].flush()

    rank_best_acc = lfs.get_best_acc_rank(best_acc)
    if rank == 0:
        txt_path = os.path.join(args.save_path, 'acc.txt')
        with open(txt_path, 'a') as f:
            save_str = "task: {}  best_acc: {:.2f}\n".format(args.task, rank_best_acc)
            f.write(save_str)



def train(train_source_iter: ContinuousDataloader, train_target_iter: ContinuousDataloader, model, optimizer: SGD, feat_adnet, pred_adnet, lfs, lr_sheduler: LrScheduler, epoch: int, args: argparse.Namespace, device):
    curr_time = get_current_time()
    print("\nNow this epoch start at :", curr_time)
    model.train()
    feat_adnet.train()
    pred_adnet.train()
    max_iters = args.iters_per_epoch * args.epochs
    for i in range(args.iters_per_epoch):

        current_iter = i + args.iters_per_epoch * epoch
        rho = current_iter/max_iters
        if args.pred_process:
            rho_pad = rho * args.rho_pad
        else:
            rho_pad = 1.0
        rho_fad = args.rho_fad
        rho_ort =  args.rho_ort + rho * (1 - args.rho_ort)  
        if 'ort' not in args.method:
            rho_ort = 0

        lr_sheduler.step()
        x_s, labels_s = next(train_source_iter)
        x_t, _ = next(train_target_iter)
        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        # get features and logit outputs
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        softmax_out = nn.Softmax(dim=1)(y)
        y_s, y_t = y.chunk(2, dim=0)

        domain_s = torch.zeros_like(labels_s)
        domain_t = torch.ones_like(labels_s)
        domain_label = torch.cat((domain_s, domain_t))

        if args.way == 'E':
            entropy = NetZoo.Entropy(softmax_out)
        else:
            entropy = None

        feat_adv_loss = lfs.dann_feat_loss(f, domain_label, feat_adnet)
        ort_loss = lfs.oploss_forward(f, y_t, labels_s, device)
        cls_loss = F.cross_entropy(y_s, labels_s)

        pred_adv_loss = lfs.adv_loss_pred(pred_adnet(y), domain_label)

        total_loss = cls_loss + rho_pad * pred_adv_loss + rho_fad * feat_adv_loss + rho_ort * ort_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print("Epoch: [{:02d}][{:03d}/{}]	total_loss:{:.3f}   cls_loss:{:.3f}	 pred_adv_loss:{:.3f}  feat_adv_loss:{:.3f}  oploss:{:.3f}".format\
                (epoch+1, i, args.iters_per_epoch, total_loss, cls_loss, pred_adv_loss*rho_pad, feat_adv_loss*rho_fad, rho_ort * ort_loss))


def discriminator_capacity(dataloder, model, feat_adnet, device):
    model.eval()
    feat_adnet = feat_adnet.eval()
    feat_entropy_list = []
    num = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(dataloder):
            num += 1
            images = images.to(device)
            target = target.to(device)
            output, f = model(images)
            softmax_out = nn.Softmax(dim=1)(output)
            f_adout = feat_adnet(f)
            batch_feat_entropy = binary_entropy(f_adout)
            bs_ent = normalize(batch_feat_entropy)
            feat_entropy_list.append(torch.mean(bs_ent))
        feat_en = torch.mean(torch.tensor(feat_entropy_list))
    return feat_en

def normalize(x):
    return (x-torch.min(x))/(torch.max(x) - torch.min(x) + 1e-10)

def get_mmd(train_source_iter, train_target_iter, model, device, mmd, args):
    model.eval()
    loss = []
    with torch.no_grad():
        for i in range(args.mmd_iters):
            x_s, labels_s = next(train_source_iter)
            x_t, _ = next(train_target_iter)
            x_s = x_s.to(device)
            x_t = x_t.to(device)
            x = torch.cat((x_s, x_t), dim=0)
            y, f = model(x)
            f_s, f_t = f.chunk(2, dim=0)
            loss.append(mmd(f_s, f_t))
    return torch.mean(torch.tensor(loss))

def validate(val_loader: DataLoader, model, device) -> float:
    model.eval()
    start_test = True
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            # get logit outputs
            output, _ = model(images)
            if start_test:
                all_output = output.float()
                all_label = target.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, output.float()), 0)
                all_label = torch.cat((all_label, target.float()), 0)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        accuracy = accuracy * 100.0
    return accuracy





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoLoss search for Domain Adaptation' )
    # parser.add_argument('--rank', type=int, default=0)
    # parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('rank', type=int)
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--arch', type=str, default='resnet50', choices=['resnet50', 'resnet101'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")

    parser.add_argument('--way', type=str, default='E', choices=['E', 'NO-ENT'])
    parser.add_argument('--method', type=str, default='Dp_ort', choices=['Dp_ort', 'Dp'])
    parser.add_argument('--description', type=str, default=' ' )

    parser.add_argument('--dset', type=str, default='office', choices=['office', 'home', 'domainnet', 'birds'], help="The dataset used")
    parser.add_argument('--task', type=str, default='D2A', help="task source 2 target")

    parser.add_argument('--save_path', type=str, default='/remote-home/share/47/meizhen/code/work_1/lsda_csvt_copy/log/search/cdan/31', help="output directory of logs")
    parser.add_argument('--data_path', type=str, default='/remote-home/source/47/meizhen/dataset/office31/office')
    parser.add_argument('--pred_process', default=False, action='store_true')
    
    
    parser.add_argument('--rho_fad', type=float, default=1.0)
    parser.add_argument('--rho_pad', type=float, default=1.0)
    parser.add_argument('--rho_ort',  type=float, default=1.0)
    parser.add_argument('--threshold', type=float, default=0.8, help="threshold for pseudo label selecting")

    parser.add_argument('--sm', type=float, default=4.0, help="threshold for pseudo label selecting")
    parser.add_argument('--port', type=str, default="12453", help="learning rate")
    parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--iters-per-epoch', default=500, type=int, help='Number of iterations per epoch')
    parser.add_argument('--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--batch-size', default=36, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', default=0.01, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', default=1e-3, type=float, metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('--center_crop', default=False, action='store_true')

    parser.add_argument('--t', default=0.35, type=int)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)
    my_rank = dist.get_rank()


    args.output_dir = osp.join(args.save_path, args.task + '/' + str(my_rank))
    config = {}
    if osp.exists(args.output_dir):
        # shutil.rmtree(args.output_dir)
        pass
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    config["out_file"] = open(osp.join(args.output_dir, get_current_time() + ".txt"), "w")

    if args.dset == "office":
        args.num_classes = 31
        data_dic = {'D':'dslr_new.txt', 'W':'webcam_new.txt', 'A':'amazon_new.txt'}

    elif args.dset == "home":
        args.num_classes = 65
        data_dic = {'C':'clipart.txt', 'P':'product.txt', 'A':'art.txt', 'R': 'real_world.txt'}
        args.data_path = '/remote-home/share/47/meizhen/dataset/office_home/list'

    elif args.dset == "birds":
        args.num_classes = 31
        data_dic = {'I':'i.txt', 'N':'n.txt', 'C':'c.txt'}

    elif args.dset == "domainnet":
        args.num_classes = 345
        data_dic = {'c':'clp.txt', 'i':'inf.txt', 'p':'pnt.txt',
                    'q':'qdr.txt', 'r':'rel.txt', 's':'skt.txt'}
        args.data_path = '/remote-home/source/47/meizhen/dataset/domainnet'

    if args.dset == "domainnet":
        args.s_dset_path = args.data_path +'/train/' + data_dic[args.task[0]]
        args.t_dset_path = args.data_path +'/test/' + data_dic[args.task[-1]]
        args.val_path    = args.data_path +'/test/' + data_dic[args.task[-1]]
    else:
        args.s_dset_path = osp.join(args.data_path, data_dic[args.task[0]])
        args.t_dset_path = osp.join(args.data_path, data_dic[args.task[-1]])
        args.val_path    = osp.join(args.data_path, data_dic[args.task[-1]])
        
    for arg in vars(args):
        print("{} = {}".format(arg, getattr(args, arg)))
        config["out_file"].write(str("{} = {}".format(arg, getattr(args, arg))) + "\n")
    config["out_file"].flush()

    main(args, config, my_rank)

    dist.destroy_process_group()