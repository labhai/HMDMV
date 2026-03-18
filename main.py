import os
import argparse
import time
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

# Dataset Module
from dataset.hotels8k import HotelsDataset
# Networks Module
from networks.hmdmv import AllCombMultiImage
from process.train import train, validation, test
# Loss & utils fuction
from loss.hmd_loss import HierarchicalMutualDisillationLoss
from utils import str2bool, set_optimizer, EarlyStopper


# Fixed All seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_option():
    parser = argparse.ArgumentParser(description='argument for training')
    parser.add_argument('--device', type=int, default=0, help='GPU number')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_view', type=int, default=2, help='number of views')
    parser.add_argument('--num_classes', type=int, default=7774, help='number of classes')
    parser.add_argument('--dataset', type=str, default='hotels8k', help='selected_dataset') # selected training data
    
    # dataset
    parser.add_argument('--data_root', type=str, default='data/hotels8k', help='dataset root path')
    parser.add_argument('--train_csv', type=str, default='train.csv', help='train csv file name')
    parser.add_argument('--val_csv', type=str, default='val.csv', help='validation csv file name')
    parser.add_argument('--test_csv', type=str, default='test.csv', help='test csv file name')

    # model architecture
    parser.add_argument('--method', type=str, default='HMDMV', help='selected method')
    parser.add_argument('--model_name', type=str, default='vit_small_r26_s32_224', help='model name')
    parser.add_argument('--hmd_loss', type=str2bool, default=True, help='hmd_loss use or not')

    # hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--patience', type=int, default=5, help='number of the patience')

    # optimization
    parser.add_argument('--optim', type=str, default='SGD', help='optimizers')
    parser.add_argument('--warmup_steps', type=int, default=5, help='warmup steps')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    
    # hyperparameters
    parser.add_argument('--temp', type=float, default=4.0, help='mutual distillation temperature')
    parser.add_argument('--lambda_param', type=float, default=0.1, help='mutual distillation lambda')
    parser.add_argument('--alpha', type=float, default=1.2, help='k-view weighted exp in HMD Loss')
    parser.add_argument('--grad_clip_norm', type=float, default=80., help='grad clip norm value')
    
    opt = parser.parse_args()

    return opt

def dataset_loader(opt):
    if opt.dataset == 'hotels8k':
        data_dir = opt.data_root
        train_dataset = HotelsDataset(data_dir=data_dir, csv_file=opt.train_csv, train=True, n=opt.num_view)
        val_dataset = HotelsDataset(data_dir=data_dir, csv_file=opt.val_csv, train=False, n=opt.num_view, classes=train_dataset.classes)
        test_dataset = HotelsDataset(data_dir=data_dir, csv_file=opt.test_csv, train=False, n=opt.num_view, classes=train_dataset.classes)
        
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=opt.num_workers, drop_last=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=opt.num_workers, drop_last=False, pin_memory=True)
        
    return train_loader, val_loader, test_loader

def set_model(opt, device):
    loss = None
    model = AllCombMultiImage(arch=opt.model_name, num_classes=opt.num_classes, num_view=opt.num_view).to(device)
    
    if opt.hmd_loss:
        loss = HierarchicalMutualDisillationLoss(dataset=opt.dataset, base_temp=opt.temp, base_lambda=opt.lambda_param, alpha=opt.alpha)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    return model, criterion, loss

def train_process(opt, train_loader, val_loader, model, criterion, hmd_loss, optimizer, scheduler, device):
    save_pth = f'result/{opt.method}/save_pth/{opt.model_name}_seed({opt.seed})_{opt.batch_size}_{opt.optim}_{opt.learning_rate}_{opt.num_view}view_temp_{opt.temp}_lambda_{opt.lambda_param}_hmd_loss_{opt.hmd_loss}.pth'
    early_stopper = EarlyStopper(patience=getattr(opt, "patience", 5), use_loss_when_tie=True, save_path=save_pth)
    best_top1_acc = 0.0
    
    for epoch in range(1, opt.epochs + 1):
        start_time = time.time()
        
        train_metrics = train(opt, model, criterion, hmd_loss, optimizer, train_loader, scheduler, device, grad_clip=opt.grad_clip_norm)
        val_metrics = validation(opt, model, criterion, optimizer, val_loader, opt.num_classes, device)
        if opt.optim != 'SGDScheduleFree' and opt.optim != 'AdamWScheduleFree':
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        end_time = time.time() 
        
        print(f"Epoch {epoch}/{opt.epochs} - Train Acc: {train_metrics['train_acc']:.4f}, Train Loss: {train_metrics['train_loss']:.4f}, "
            f"Top-1 Acc: {val_metrics['top1_acc']:.4f}, Top-5 Acc: {val_metrics['top5_acc']:.4f}, Val Loss: {val_metrics['loss']:.4f}, lr: {current_lr:.6f}, time: {end_time - start_time:.2f} seconds")
            
        # === EarlyStopping Check ===
        val_acc = val_metrics['top1_acc']
        val_loss = val_metrics['loss']
        
        best_improved, should_stop = early_stopper.check(metric=val_acc, model=model, tiebreak_loss=val_loss)
        
        if best_improved:
            if val_acc > best_top1_acc:
                best_top1_acc = val_acc
        
        if should_stop:
            print("Early stopping triggered. Stopping training...")
            break

    return train_metrics, val_metrics

def main():
    opt = parse_option()
    set_seed(opt.seed)
    os.makedirs(f"result/{opt.method}/save_pth", exist_ok=True)
    os.makedirs(f"result/{opt.method}/txt", exist_ok=True)
    os.makedirs(f"result/{opt.method}/csv_file", exist_ok=True)
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")

    # Training, Validation
    train_loader, val_loader, test_loader = dataset_loader(opt)
    
    model, criterion, hmd_loss = set_model(opt, device)
    optimizer = set_optimizer(opt, model)
    if opt.optim == "AdamWScheduleFree" or opt.optim == "SGDScheduleFree":
        scheduler = None
    else:
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=opt.epochs // 2, cycle_mult=1.0, max_lr=opt.learning_rate, min_lr=1e-6, warmup_steps=opt.warmup_steps, gamma=0.5)

    train_process(opt, train_loader, val_loader, model, criterion, hmd_loss, optimizer, scheduler, device)

    ckpt_path = f'result/{opt.method}/save_pth/{opt.model_name}_seed({opt.seed})_{opt.batch_size}_{opt.optim}_{opt.learning_rate}_{opt.num_view}view_temp_{opt.temp}_lambda_{opt.lambda_param}_hmd_loss_{opt.hmd_loss}.pth'
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    test_metrics = test(opt, model, criterion, optimizer, test_loader, opt.num_classes, device)

    for view_type, metrics in test_metrics.items():
        print(f"  {view_type}: Top-1 Acc = {metrics['top1_acc']:.2f}% / Top-5 Acc = {metrics['top5_acc']:.2f}% / Loss = {metrics['loss']:.4f}")
    
    # Test Result save (txt)
    results_filename = f'result/{opt.method}/txt/{opt.model_name}_{opt.batch_size}_{opt.optim}_{opt.learning_rate}_hmd_loss_{opt.hmd_loss}_results_summary.txt'

    with open(results_filename, 'a') as file:
        file.write(f"(device: {opt.device}, Seed: {opt.seed}, HMDMV: {opt.hmd_loss}, temp: {opt.temp}, lambda: {opt.lambda_param})\n")
        file.write("Performance Across All Views:\n")
        
        for view_type, metrics in test_metrics.items():
            file.write(f"{view_type}:\n")
            file.write(f"  Top-1 Acc: {metrics['top1_acc']:.4f}\n")
            file.write(f"  Top-5 Acc: {metrics['top5_acc']:.4f}\n")
            file.write(f"  Loss: {metrics['loss']:.4f}\n")
            file.write("\n")
    
if __name__=='__main__':
    main()
