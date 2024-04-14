import configargparse
from torch._C import default_generator
import data_loader
import os
import torch
import models
import utils
from utils import str2bool
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# torch.set_num_threads(10)
#############################################################################
#在head、特征层和clc上都加入loss
#############################################################################

def get_parser():
    """ get parser  """
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add("--seed", type=int, default=0)

    # data loading
    parser.add_argument('--data_dir', type=str, default='./data/ERN')
    parser.add_argument('--src_domain', type=str, default='s0')
    parser.add_argument('--tgt_domain', type=str, default='s1')

    # network
    parser.add_argument("--backbone", type=str, default='eegtransfernet')

    # training
    parser.add_argument("--batch_size", type=int, default=170)
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False, help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")

    # optimizer
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=False)

    # transfer
    parser.add_argument('--transfer_loss_weight', type=float, default=1)
    parser.add_argument('--head_loss_weight', type=float, default=1)
    parser.add_argument('--transfer_loss', type=str, default='adv')
    return parser   

def set_random_seed(seed=0):
    """ set seed """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_dataloader(args):
    """ get dataloader """
    folder_src = os.path.join(args.data_dir, args.src_domain)
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    setattr(args, "folder_src", folder_src)
    src_dataloader, n_class, _, _, loss_weight = data_loader.load_data(folder_src, args.batch_size, False, args.num_workers)
    tgt_train_dataloader, n_class, _, _, _ = data_loader.load_data(folder_tgt, args.batch_size, False, args.num_workers)
    tgt_test_dataloader, n_class, _, _, _ = data_loader.load_data(folder_tgt, args.batch_size, False, args.num_workers)
    return src_dataloader, tgt_train_dataloader, tgt_test_dataloader, n_class, loss_weight

def get_model(args):
    model = models.EEGTransferNet_CA( # EEGTransferNet_frozenBackbone_CA
        args.n_class, transfer_loss=args.transfer_loss, base_net = args.backbone, max_iter=args.max_iter, folder_src=args.folder_src, batch_size=args.batch_size, num_workers=args.num_workers, loss_weight = args.loss_weight).to(args.device)
    return model

def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    # optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    return optimizer

def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler

def test(model, target_test_loader, args):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    All_pred = torch.tensor([]).cuda()
    T_target = torch.tensor([]).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(args.device), target.to(args.device)
            s_output = model.predict(data)
            T_target = torch.cat((T_target, target))
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            All_pred = torch.cat((All_pred, pred))
            correct += torch.sum(pred == target)
    AUC_score = roc_auc_score(T_target.cpu(), All_pred.cpu())
    acc = 100. * correct / len_target_dataset
    return acc, test_loss.avg, AUC_score

def train(source_loader, target_train_loader, target_test_loader, model, optimizer, lr_scheduler, args):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch    

    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    best_auc = 0
    stop = 0
    log = []

    start = datetime.datetime.now()
    for e in range(1, args.n_epoch+1):
        model.train()
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.epoch_based_processing(n_batch) # for daan only

        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)

        for _ in range(n_batch):
            data_source, label_source = next(iter_source)
            data_target, _ = next(iter_target)   
            data_source, label_source = data_source.to(args.device), label_source.to(args.device)
            data_target = data_target.to(args.device)   

            clf_loss, transfer_loss = model(data_source, data_target, label_source)
            loss = clf_loss + args.transfer_loss_weight * transfer_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())            

        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg])

        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}'.format(
                        e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg)


        # Test
        stop += 1
        test_acc, test_loss, AUC_score = test(model, target_test_loader, args)
        info += ', test_loss {:4f}, test_acc: {:.4f}, AUC: {:.4f}'.format(test_loss, test_acc, AUC_score)
        np_log = np.array(log) #, dtype=float
        np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')
        if best_auc < AUC_score:
            best_auc = AUC_score
            stop = 0
        if args.early_stop > 0 and stop >= args.early_stop:
            print(info)
            break
        print(info)
    end = datetime.datetime.now()
    print(end-start)

    print('Transfer result: {:.4f}'.format(best_auc))         
    f = open('./EEGTransferNet_ERN/EEGTransferNet_ERN_AccLog.txt', 'a')
    f.write('%f ' % best_auc)  

def main():
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(args)
    set_random_seed(args.seed)
    src_dataloader, tgt_train_dataloader, tgt_test_dataloader, n_class, loss_weight = get_dataloader(args)
    setattr(args, "n_class", n_class)
    setattr(args, "loss_weight", loss_weight)
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(src_dataloader), len(tgt_train_dataloader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    model = get_model(args)
    optimizer = get_optimizer(model, args)

    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None
    train(src_dataloader, tgt_train_dataloader, tgt_test_dataloader, model, optimizer, scheduler, args)



if __name__ == "__main__":
    main()

