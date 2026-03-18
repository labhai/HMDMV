import torch
import pandas as pd
import einops
from tqdm import tqdm
import torch.nn.functional as F
from itertools import combinations
from collections import defaultdict

def compute_ensemble_accuracy_topk(grouped_probs, grouped_gt, k=1):
    correct = 0
    total = 0
    for hotel_id, prob_list in grouped_probs.items():
        avg_prob = torch.mean(torch.stack(prob_list), axis=0)
        if k == 1:
            final_pred = torch.argmax(avg_prob)
            if final_pred.item() == grouped_gt[hotel_id].item():
                correct += 1
        else:
            topk_preds = torch.argsort(avg_prob, descending=True)[:k]
            if grouped_gt[hotel_id].item() in topk_preds.tolist():
                correct += 1
        total += 1
    accuracy_k = (correct / total) * 100
    return accuracy_k

def train(opt, model, criterion, hmd_loss, optimizer, train_loader, scheduler, device, grad_clip=None):
    model.train()
    if opt.optim == 'SGDScheduleFree' or opt.optim == 'AdamWScheduleFree':
        optimizer.train()
        
    train_acc = {}
    total_loss = 0

    for idx, (images, targets, _, _) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        B, N, _, _, _ = images.shape
        images = images.to(device)
        targets = targets.long().to(device)

        output = model(images)
        batch_loss = torch.tensor(0.).to(device, non_blocking=True)

        for cur_view, view_type in enumerate(output):
            cur_view = cur_view + 1
            logits = output[view_type]

            if cur_view == 1:
                t = targets.flatten()
            else:
                NC = len(list(combinations(range(opt.num_view), cur_view)))
                t = targets[:, 0].repeat_interleave(NC)   # [B*NC]

            base_loss = criterion(logits, t)
            batch_loss += base_loss

            pred = torch.argmax(logits, dim=1)

            if view_type not in train_acc:
                train_acc[view_type] = {'total': 0, 'correct': 0}

            pred = pred[t != -1]
            t = t[t != -1]

            train_acc[view_type]['correct'] += (pred == t).long().sum().item()
            train_acc[view_type]['total'] += t.numel()

        '''
        Hierarchical method: 
        Step 1: 1_view -> full_view
        Step 2: 2_view -> full_view
        ...
        Step (N-1): (N-1)_view -> full_view
        '''
        if hmd_loss is not None:
            full_view = f'{opt.num_view}_view'  # full multi-view
            full_view_logits = output[full_view]

            # Hierarchical Mutual Distillation
            for cur_view in range(1, opt.num_view):
                prev_view = f'{cur_view}_view'
                NC = output[prev_view].size(0) // B  # num_combinations

                if prev_view in output:
                    prev_view_logits = einops.rearrange(output[prev_view], '(b nc) k -> b nc k', b=B, nc=NC)
                    mutual_distillation_loss = hmd_loss(full_view_logits, prev_view_logits, targets[:, 0], cur_view, opt.num_view)
                    batch_loss += mutual_distillation_loss

        optimizer.zero_grad()
        batch_loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += batch_loss.item()

    metrics_dict = {}
    for view_type in train_acc:
        if train_acc[view_type]['total'] > 0:
            acc = train_acc[view_type]['correct'] / train_acc[view_type]['total'] * 100  # 정확도 계산
            metrics_dict[view_type] = acc
    
    train_loss = total_loss / len(train_loader)
    mv_view = f'{opt.num_view}_view'

    return {
        'train_loss': train_loss,
        'train_acc': metrics_dict[mv_view]
    }

def validation(opt, model, criterion, optimizer, val_loader, num_classes, device):
    model.eval()
    if opt.optim == 'SGDScheduleFree' or opt.optim == 'AdamWScheduleFree':
        optimizer.eval()
        
    total_loss = 0.0

    mv_view = f'{opt.num_view}_view'
    results_dict = {mv_view: {'logits': [], 'classes': [], 'preds': [], 'probs': [], 'ids': []}}
    
    with torch.no_grad():
        for images, targets, hotel_ids, _ in tqdm(val_loader, desc="Validation", leave=False):
            B, N, _, _, _ = images.shape  # multi-view
            images = images.to(device)
            targets = targets.long().to(device)

            output = model.predict(images)

            logits = output
            t = targets[:, 0].flatten()
            
            batch_loss = criterion(logits, t)
            probs = F.softmax(logits, dim=1)
                
            total_loss += batch_loss.item()

            results_dict[mv_view]['logits'].append(logits.cpu())
            results_dict[mv_view]['classes'].append(t.cpu())
            results_dict[mv_view]['probs'].append(probs.cpu())
            results_dict[mv_view]['ids'].extend(hotel_ids)

    for view_type in results_dict:
        results_dict[mv_view]['logits'] = torch.cat(results_dict[view_type]['logits'])
        results_dict[mv_view]['classes'] = torch.cat(results_dict[view_type]['classes'])
        results_dict[mv_view]['probs'] = torch.cat(results_dict[view_type]['probs'])

    metrics_dict = {}

    gt_classes = results_dict[mv_view]['classes'].to(device)
    probs = results_dict[mv_view]['probs'].to(device)
    ids = results_dict[mv_view]['ids']

    # Each classes probability collection (all combination datasets)
    grouped_probs = defaultdict(list)
    grouped_gt = {}
        
    for i, hotel_id in enumerate(ids):
        hotel_id = str(hotel_id.item()).strip()
        grouped_probs[hotel_id].append(probs[i])
        grouped_gt[hotel_id] = gt_classes[i]
        
    top1_acc = compute_ensemble_accuracy_topk(grouped_probs, grouped_gt, k=1)
    top5_acc = compute_ensemble_accuracy_topk(grouped_probs, grouped_gt, k=5)

    metrics_dict[mv_view] = {
        'top1_acc': top1_acc,
        'top5_acc': top5_acc if top5_acc is not None else None,
    }

    val_loss = total_loss / len(val_loader)

    mv_metrics = metrics_dict[mv_view]
    mv_metrics['loss'] = val_loss

    return mv_metrics

def test(opt, model, criterion, optimizer, test_loader, num_classes, device):
    model.eval()
    if opt.optim == 'SGDScheduleFree' or opt.optim == 'AdamWScheduleFree':
        optimizer.eval()
        
    total_loss = 0

    mv_view = f'{opt.num_view}_view'
    results_dict = {mv_view: {'logits': [], 'classes': [], 'probs': [], 'ids': []}}
    
    with torch.no_grad():
        for images, targets, hotels_ids, _ in tqdm(test_loader, desc="Test", leave=False):
            B, N, _, _, _ = images.shape  # multi-view
            images = images.to(device)
            targets = targets.long().to(device)
            
            output = model.predict(images)
            logits = output
            
            t = targets[:, 0].flatten()

            batch_loss = criterion(logits, t)
            probs = F.softmax(logits, dim=1)

            total_loss += batch_loss.item()

            results_dict[mv_view]['logits'].append(logits.cpu())
            results_dict[mv_view]['classes'].append(t.cpu())
            results_dict[mv_view]['probs'].append(probs.cpu())
            results_dict[mv_view]['ids'].extend(hotels_ids)


    results_dict[mv_view]['logits'] = torch.cat(results_dict[mv_view]['logits'])
    results_dict[mv_view]['classes'] = torch.cat(results_dict[mv_view]['classes'])
    results_dict[mv_view]['probs'] = torch.cat(results_dict[mv_view]['probs'])

    metrics_dict = {}
    
    gt_classes = results_dict[mv_view]['classes'].to(device)
    probs = results_dict[mv_view]['probs'].to(device)
    ids = results_dict[mv_view]['ids']

    # Each classes probability collection (all combination datasets)
    grouped_probs = defaultdict(list)
    grouped_gt = {}
        
    for i, hotel_id in enumerate(ids):
        hotel_id = str(hotel_id.item()).strip()
        grouped_probs[hotel_id].append(probs[i])
        grouped_gt[hotel_id] = gt_classes[i]
            
    # print("Grouped hotel IDs and number of samples per group:")
    # for hotel_id, prob_list in grouped_probs.items():
    #     print(f"Hotel ID: {hotel_id} -> {len(prob_list)} samples")
        
    top1_acc = compute_ensemble_accuracy_topk(grouped_probs, grouped_gt, k=1)
    top5_acc = compute_ensemble_accuracy_topk(grouped_probs, grouped_gt, k=5)
    
    csv_data = []
    for hotel_id, prob_list in grouped_probs.items():
        avg_prob = torch.mean(torch.stack(prob_list), axis=0)
        final_pred = torch.argmax(avg_prob).item()
        target_val = grouped_gt[hotel_id].item()
                
        top5_preds = avg_prob.topk(5).indices.tolist()
        top1_prob = avg_prob[final_pred].item()
                
        csv_data.append({
            'id': hotel_id,
            'target': target_val,
            'top1_pred': final_pred,
            'top5_preds': top5_preds,
            'top1_prob': top1_prob
        })
            
    csv_filename = f"result/{opt.method}/csv_file/test_results_{mv_view}_{opt.model_name}_{opt.batch_size}_{opt.optim}_{opt.learning_rate}_{opt.num_view}view_temp_{opt.temp}_hmd_loss_{opt.hmd_loss}.csv"
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_filename, index=False)
    print(f"Saved ensemble results for {mv_view} to {csv_filename}")
        
    metrics_dict[mv_view] = {
        'top1_acc': top1_acc,
        'top5_acc': top5_acc if top5_acc is not None else None,
    }
    
    test_loss = total_loss / len(test_loader)
    
    metrics_dict[mv_view]['loss'] = test_loss

    return metrics_dict