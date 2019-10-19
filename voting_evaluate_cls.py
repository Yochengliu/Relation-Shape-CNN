import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
from torchvision import transforms
from models import RSCNN_SSN_Cls as RSCNN_SSN
from data import ModelNet40Cls
import utils.pytorch_utils as pt_utils
import utils.pointnet2_utils as pointnet2_utils
import data.data_utils as d_utils
import argparse
import random
import yaml

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed) 

parser = argparse.ArgumentParser(description='Relation-Shape CNN Shape Classification Voting Evaluation')
parser.add_argument('--config', default='cfgs/config_ssn_cls.yaml', type=str)

NUM_REPEAT = 300
NUM_VOTE = 10

def main():
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    for k, v in config['common'].items():
        setattr(args, k, v)
    
    test_transforms = transforms.Compose([
        d_utils.PointcloudToTensor()
    ])

    test_dataset = ModelNet40Cls(num_points = args.num_points, root = args.data_root, transforms=test_transforms, train=False)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=int(args.workers), 
        pin_memory=True
    )
    
    model = RSCNN_SSN(num_classes = args.num_classes, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True)
    model.cuda()
    
    if args.checkpoint is not '':
        model.load_state_dict(torch.load(args.checkpoint))
        print('Load model successfully: %s' % (args.checkpoint))
    
    # evaluate
    PointcloudScale = d_utils.PointcloudScale()   # initialize random scaling
    model.eval()
    global_acc = 0
    for i in range(NUM_REPEAT):
        preds = []
        labels = []
        for j, data in enumerate(test_dataloader, 0):
            points, target = data
            points, target = points.cuda(), target.cuda()
            points, target = Variable(points, volatile=True), Variable(target, volatile=True)
            
            # fastest point sampling
            fps_idx = pointnet2_utils.furthest_point_sample(points, 1200)  # (B, npoint)
            pred = 0
            for v in range(NUM_VOTE):
                new_fps_idx = fps_idx[:, np.random.choice(1200, args.num_points, False)]
                new_points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), new_fps_idx).transpose(1, 2).contiguous()
                if v > 0:
                    new_points.data = PointcloudScale(new_points.data)
                pred += F.softmax(model(new_points), dim = 1)
            pred /= NUM_VOTE
            target = target.view(-1)
            _, pred_choice = torch.max(pred.data, -1)
            
            preds.append(pred_choice)
            labels.append(target.data)
    
        preds = torch.cat(preds, 0)
        labels = torch.cat(labels, 0)
        acc = (preds == labels).sum() / labels.numel()
        if acc > global_acc:
            global_acc = acc
        print('Repeat %3d \t Acc: %0.6f' % (i + 1, acc))
    print('\nBest voting acc: %0.6f' % (global_acc))
        
if __name__ == '__main__':
    main()