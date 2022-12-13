import os
import numpy as np
import torch
from network import ImageClassifier
from torch.utils.data import DataLoader
from data_list import PseudoList, ImageList
import torchvision.transforms as transforms
from transforms import ResizeImage
import torch.nn.functional as F
import backbone as BackboneNetwork
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def make_new_list(src_path, tar_path, model_path, save_path, backbone, my_rank):
    device = torch.device("cuda:%d" % my_rank)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        ResizeImage(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        normalize])


    train_source_dataset = ImageList(open(tar_path).readlines(), transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=36, shuffle=False, num_workers=4, drop_last=False)

    model = ImageClassifier(backbone, 31)

    check_point = torch.load(model_path, map_location=device)

    new_check = {}
    for k in check_point['backbone'].keys():
        if k.startswith('module'):
            a = k.split('.',1)[1]
            new_check[a] = check_point['backbone'][k]

    model.load_state_dict(new_check)
    model = model.to(device)
    pred_outputs = torch.LongTensor([])
    pseu_labels = torch.LongTensor([])
    with torch.no_grad():
        for data in train_source_loader:
            input = data[0]
            input = input.to(device)
            outputs, _ = model(input)
            pred_outputs = torch.cat([pred_outputs, outputs.cpu()], dim=0)
            pseu_labels = torch.cat([pseu_labels, torch.argmax(outputs.cpu(), dim=1)], dim=0)

    pred_soft = F.softmax(pred_outputs, dim=1)
    max_prob, max_label = torch.max(pred_soft, dim=1)
    good_id  = (max_prob>0.95).nonzero()

    lists = open(tar_path).readlines()
    pseudo_tar_list = [lists[good_id[i]].split(' ')[0] + ' ' + str(pseu_labels[i].item()) for i in range(len(good_id))]
    src_list = open(src_path).readlines()

    fw = open(save_path, 'w')
    for l in pseudo_tar_list:
        fw.write(l)
        fw.write('\n')
    for ll in src_list:
        fw.write(ll)

#
# backbone = BackboneNetwork.__dict__['resnet50'](pretrained=True)
# make_new_list('/remote-home/share/47/meizhen/dataset/office31/share_office/dslr_new.txt', \
#               '/remote-home/share/47/meizhen/dataset/office31/share_office/amazon_new.txt',\
#               '/remote-home/meizhen/10.26.2.244/LSDA-main/log/0/from0.pth', \
#               '/remote-home/meizhen/10.26.2.244/LSDA-main/log/0/pseudo_src_list.txt',\
#               backbone, 0)





