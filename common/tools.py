import torch

def get_center_feature(val_loader, model, device, cls_num) -> float:
    # switch to evaluate mode
    model.eval()
    start_test = True
    c_f_all = []
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            output, f = model(images)
            if start_test:
                all_f = f.float()
                all_label = target.float()
                start_test = False
            else:
                all_f = torch.cat((all_f, f.float()), 0)
                all_label = torch.cat((all_label, target.float()), 0)
        for j in range(cls_num):
            f = all_f[all_label == j]
            center_f = torch.mean(f, dim=0)
            c_f_all.append(center_f.unsqueeze(0))
    c_f_all = torch.cat(c_f_all, dim=0)
    return c_f_all