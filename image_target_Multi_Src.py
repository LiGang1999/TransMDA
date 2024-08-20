import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList_idx
import random
from scipy.spatial.distance import cdist
from loss import KnowledgeDistillationLoss
from datasets.DomainNet import get_domainnet_dloader


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


def data_load(args):
    # target: for training;
    # target_: for computing centroid;
    # test: for testing.

    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    if args.dset == 'DomainNet':
        dset_loaders["target"], _ = get_domainnet_dloader(args.base_path,
                                                        args.domains[args.t],
                                                        args.batch_size,
                                                        args.worker,
                                                        Isshuffle=True)
        dset_loaders["target_"], dset_loaders["test"] = get_domainnet_dloader(args.base_path,
                                                        args.domains[args.t],
                                                        args.batch_size * 3,
                                                        args.worker,
                                                        Isshuffle=False)
        return dset_loaders
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF_list = [network.ResBase(res_name=args.net, se=args.se, nl=args.nl).cuda() for i in range(len(args.src))]
    elif args.net[0:3] == 'vgg':
        netF_list = [network.VGGBase(vgg_name=args.net).cuda() for i in range(len(args.src))]
    elif args.net == 'vit':
        netF_list = [network.ViT().cuda() for i in range(len(args.src))]

    w = torch.zeros((len(args.src),))
    print(w)

    netB_list = [network.feat_bootleneck(type=args.classifier, feature_dim=netF_list[i].in_features,
                                   bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))]
    netC_list = [network.feat_classifier(type=args.layer, class_num=args.class_num,
                                   bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))]
    netG_list = [network.scalar(w[i]).cuda() for i in range(len(args.src))]

    ### add teacher module
    if args.net[0:3] == 'res':
        netF_t_list = [network.ResBase(res_name=args.net, se=args.se, nl=args.nl).cuda() for i in range(len(args.src))]
    elif args.net[0:3] == 'vgg':
        netF_t_list = [network.VGGBase(vgg_name=args.net).cuda() for i in range(len(args.src))]
    elif args.net == 'vit':
        netF_t_list = [network.ViT().cuda() for i in range(len(args.src))]
    netB_t_list = [network.feat_bootleneck(type=args.classifier, feature_dim=netF_t_list[i].in_features,
                                   bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))]

    param_group = []
    for i in range(len(args.src)):
        modelpath = args.output_dir_src[i] + '/source_F.pt'
        print(modelpath)
        netF_list[i].load_state_dict(torch.load(modelpath))
        netF_list[i].eval()
        for _, v in netF_list[i].named_parameters():
            if args.lr_decay1 > 0:
                param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
            else:
                v.requires_grad = False

        modelpath = args.output_dir_src[i] + '/source_B.pt'
        print(modelpath)
        netB_list[i].load_state_dict(torch.load(modelpath))
        netB_list[i].eval()
        for _, v in netB_list[i].named_parameters():
            if args.lr_decay2 > 0:
                param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
            else:
                v.requires_grad = False

        modelpath = args.output_dir_src[i] + '/source_C.pt'
        print(modelpath)
        netC_list[i].load_state_dict(torch.load(modelpath))
        netC_list[i].eval()
        for _, v in netC_list[i].named_parameters():
            v.requires_grad = False

        for _, v in netG_list[i].named_parameters():
            param_group += [{'params':v, 'lr':args.lr}]
            # param_group += [{'params':v, 'lr':args.lr * 1e1}]# learning rate for domain weight!!!

        ### initial from student
        netF_t_list[i].load_state_dict(netF_list[i].state_dict())
        netB_t_list[i].load_state_dict(netB_list[i].state_dict())

        ### remove grad
        for _, v in netF_t_list[i].named_parameters():
            v.requires_grad = False
        for _, v in netB_t_list[i].named_parameters():
            v.requires_grad = False

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    # max_iter = 5 * 1250# DomainNet
    # interval_iter = 1250# DomainNet
    iter_num = 0

    while iter_num < max_iter:
        try:
            # inputs_test, _, tar_idx = iter_test.next()
            inputs_test, _, tar_idx = iter_test.__next__()
        except:
            iter_test = iter(dset_loaders["target"])
            # inputs_test, _, tar_idx = iter_test.next()
            inputs_test, _, tar_idx = iter_test.__next__()

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and args.cls_par > 0:
            initc = []
            all_feas = []
            for i in range(len(args.src)):
                netF_list[i].eval()
                netB_list[i].eval()
                netF_t_list[i].eval()
                netB_t_list[i].eval()
                if args.dset == 'DomainNet':
                    temp1, temp2 = obtain_label(dset_loaders['target_'], netF_t_list[i], netB_t_list[i], netC_list[i])
                else:
                    temp1, temp2 = obtain_label(dset_loaders['test'], netF_t_list[i], netB_t_list[i], netC_list[i])
                temp1 = torch.from_numpy(temp1).cuda()#mem_label
                temp2 = torch.from_numpy(temp2).cuda()#dd
                initc.append(temp1)
                all_feas.append(temp2)
                netF_list[i].train()
                netB_list[i].train()

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        outputs_all = torch.zeros(len(args.src), inputs_test.shape[0], args.class_num)
        weights_all = torch.ones(inputs_test.shape[0], len(args.src))
        outputs_all_w = torch.zeros(inputs_test.shape[0], args.class_num)

        for i in range(len(args.src)):
            features_test = netB_list[i](netF_list[i](inputs_test))
            outputs_test = netC_list[i](features_test)
            weights_test = netG_list[i](features_test)
            outputs_all[i] = outputs_test
            weights_all[:, i] = weights_test.squeeze()

        z = torch.sum(weights_all, dim=1)
        z = z + 1e-16

        weights_all = torch.transpose(torch.transpose(weights_all, 0, 1) / z, 0, 1)
        outputs_all = torch.transpose(outputs_all, 0, 1)

        z_ = torch.sum(weights_all, dim=0)
        z_2 = torch.sum(weights_all)
        z_ = z_ / z_2

        for i in range(inputs_test.shape[0]):
            outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i], 0, 1), weights_all[i])

        if args.cls_par > 0:
            initc_ = torch.zeros(initc[0].size()).cuda()
            temp = all_feas[0]
            all_feas_ = torch.zeros(temp[tar_idx, :].size()).cuda()
            for i in range(len(args.src)):
                initc_ = initc_ + z_[i] * initc[i].float()# Compute centroid
                src_fea = all_feas[i]
                all_feas_ = all_feas_ + z_[i] * src_fea[tar_idx, :]
            dd = torch.cdist(all_feas_.float(), initc_.float(), p=2)
            pred_label = dd.argmin(dim=1)
            pred_label = pred_label.int()
            pred = pred_label.long()

            classifier_loss = args.cls_par * nn.CrossEntropyLoss()(outputs_all_w, pred.cpu())

            if args.pls:
                pls_loss = KnowledgeDistillationLoss(alpha=1.)(outputs_all_w, dd.cpu())
                classifier_loss += pls_loss * 10.0
                if iter_num == max_iter:
                    print('execution...')
                    print('z_:', z_)
        else:
            classifier_loss = torch.tensor(0.0)

        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_all_w)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()

        # EMA update for the teacher
        with torch.no_grad():
            m = 0.998# momentum parameter
            # m = 0.999# DomainNet
            for i in range(len(args.src)):
                for param_q, param_k in zip(netF_list[i].parameters(), netF_t_list[i].parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
                for param_q, param_k in zip(netB_list[i].parameters(), netB_t_list[i].parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            for i in range(len(args.src)):
                netF_list[i].eval()
                netB_list[i].eval()

            if args.dset == 'VISDA-C':
                print("Do not support VISDA-C"+'\n')
            else:
                acc, _ = cal_acc_multi(dset_loaders['test'], netF_list, netB_list, netC_list, netG_list, args)
                log_str = 'Iter:{}/{}; Accuracy = {:.2f}%'.format(iter_num, max_iter, acc)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            for i in range(len(args.src)):
                netF_list[i].train()
                netB_list[i].train()

    if args.issave:
        for i in range(len(args.src)):
            torch.save(netF_list[i].state_dict(), osp.join(args.output_dir, "target_F_" + str(i) + "_" + args.savename + ".pt"))
            torch.save(netB_list[i].state_dict(), osp.join(args.output_dir, "target_B_" + str(i) + "_" + args.savename + ".pt"))
            torch.save(netC_list[i].state_dict(), osp.join(args.output_dir, "target_C_" + str(i) + "_" + args.savename + ".pt"))
            torch.save(netG_list[i].state_dict(), osp.join(args.output_dir, "target_G_" + str(i) + "_" + args.savename + ".pt"))


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def obtain_label(loader, netF, netB, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            # data = iter_test.next()
            data = iter_test.__next__()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netB(netF(inputs.float()))
            outputs = netC(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])

    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for _ in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy*100, acc*100)
    print(log_str+'\n')
    #return pred_label.astype('int')
    return initc, all_fea


def cal_acc_multi(loader, netF_list, netB_list, netC_list, netG_list, args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            # data = iter_test.next()
            data = iter_test.__next__()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs_all = torch.zeros(len(args.src), inputs.shape[0], args.class_num)
            weights_all = torch.ones(inputs.shape[0], len(args.src))
            outputs_all_w = torch.zeros(inputs.shape[0], args.class_num)

            for i in range(len(args.src)):
                features = netB_list[i](netF_list[i](inputs))
                outputs = netC_list[i](features)
                weights = netG_list[i](features)
                outputs_all[i] = outputs
                weights_all[:, i] = weights.squeeze()

            z = torch.sum(weights_all, dim=1)
            z = z + 1e-16

            weights_all = torch.transpose(torch.transpose(weights_all,0,1) / z,0,1)
            print(weights_all.mean(dim=0))
            outputs_all = torch.transpose(outputs_all, 0, 1)

            for i in range(inputs.shape[0]):
                outputs_all_w[i] = torch.matmul(torch.transpose(outputs_all[i],0,1), weights_all[i])

            if start_test:
                all_output = outputs_all_w.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs_all_w.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy * 100, mean_ent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TransMDA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--t', type=int, default=0, help="target") ## Choose which domain to set as target {0 to len(names)-1}
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")#64,32,24
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'DomainNet'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='vit', help="alexnet, vgg16, resnet50, res101, vit")
    parser.add_argument('--seed', type=int, default=3407, help="random seed")

    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--pls', type=bool, default=False)
    parser.add_argument('--se', type=bool, default=False)
    parser.add_argument('--nl', type=bool, default=False)

    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)#256,512,2048
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('-bp', '--base-path', default="./")
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset == 'DomainNet':
        names = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        args.domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        args.class_num = 345

    args.src = []
    for i in range(len(names)):
        if i == args.t:
            continue
        else:
            args.src.append(names[i])

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i != args.t:
            continue
        folder = './data/'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        print(args.t_dset_path)

    args.output_dir_src = []
    for i in range(len(args.src)):
        args.output_dir_src.append(osp.join(args.output_src, args.da, args.dset, args.src[i][0].upper()))
    print(args.output_dir_src)
    args.output_dir = osp.join(args.output, args.da, args.dset, names[args.t][0].upper())
    args.name=names[args.t][0].upper()

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.savename = 'par_' + str(args.cls_par)
    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()
    train_target(args)


# office-home

# CUDA_VISIBLE_DEVICES=0 python image_target_Multi_Src.py --da uda --dset office-home --t 0 --output_src ckps/source/ --output ckps/target_Multi_Src/ --pls True > log_t0_office-home.txt

# CUDA_VISIBLE_DEVICES=1 python image_target_Multi_Src.py --da uda --dset office-home --t 1 --output_src ckps/source/ --output ckps/target_Multi_Src/ --pls True > log_t1_office-home.txt

# CUDA_VISIBLE_DEVICES=2 python image_target_Multi_Src.py --da uda --dset office-home --t 2 --output_src ckps/source/ --output ckps/target_Multi_Src/ --pls True > log_t2_office-home.txt

# CUDA_VISIBLE_DEVICES=3 python image_target_Multi_Src.py --da uda --dset office-home --t 3 --output_src ckps/source/ --output ckps/target_Multi_Src/ --pls True > log_t3_office-home.txt
