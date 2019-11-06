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
from models import RSCNN_MSN_Seg as RSCNN_MSN
from data import ShapeNetPart
import utils.pytorch_utils as pt_utils
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

parser = argparse.ArgumentParser(description='Relation-Shape CNN Shape Part Segmentation Voting Evaluate')
parser.add_argument('--config', default='cfgs/config_msn_partseg.yaml', type=str)

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
    
    test_dataset = ShapeNetPart(root = args.data_root, num_points = args.num_points, split = 'test', normalize = True, transforms = test_transforms)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=int(args.workers), 
        pin_memory=True
    )
    
    model = RSCNN_MSN(num_classes = args.num_classes, input_channels = args.input_channels, relation_prior = args.relation_prior, use_xyz = True)
    model.cuda()

    if args.checkpoint is not '':
        model.load_state_dict(torch.load(args.checkpoint))
        print('Load model successfully: %s' % (args.checkpoint))

    # evaluate
    PointcloudScale = d_utils.PointcloudScale(scale_low=0.87, scale_high=1.15)   # initialize random scaling
    model.eval()
    global_Class_mIoU, global_Inst_mIoU = 0, 0
    seg_classes = test_dataset.seg_classes
    seg_label_to_cat = {}           # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat
    
    for i in range(NUM_REPEAT):
        shape_ious = {cat:[] for cat in seg_classes.keys()}
        for _, data in enumerate(test_dataloader, 0):
            points, target, cls = data
            points, target = Variable(points, volatile=True), Variable(target, volatile=True)
            points, target = points.cuda(), target.cuda()

            batch_one_hot_cls = np.zeros((len(cls), 16))   # 16 object classes
            for b in range(len(cls)):
                batch_one_hot_cls[b, int(cls[b])] = 1
            batch_one_hot_cls = torch.from_numpy(batch_one_hot_cls)
            batch_one_hot_cls = Variable(batch_one_hot_cls.float().cuda())

            pred = 0
            new_points = Variable(torch.zeros(points.size()[0], points.size()[1], points.size()[2]).cuda(), volatile=True)
            for v in range(NUM_VOTE):
                if v > 0:
                    new_points.data = PointcloudScale(points.data)
                pred += F.softmax(model(new_points, batch_one_hot_cls), dim = 2)
            pred /= NUM_VOTE
            
            pred = pred.data.cpu()
            target = target.data.cpu()
            pred_val = torch.zeros(len(cls), args.num_points).type(torch.LongTensor)
            # pred to the groundtruth classes (selected by seg_classes[cat])
            for b in range(len(cls)):
                cat = seg_label_to_cat[target[b, 0]]
                logits = pred[b, :, :]   # (num_points, num_classes)
                pred_val[b, :] = logits[:, seg_classes[cat]].max(1)[1] + seg_classes[cat][0]
            
            for b in range(len(cls)):
                segp = pred_val[b, :]
                segl = target[b, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if torch.sum((segl == l) | (segp == l)) == 0:
                        # part is not present in this shape
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = torch.sum((segl == l) & (segp == l)) / float(torch.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))
        
        instance_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                instance_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_class_ious = np.mean(list(shape_ious.values()))
        
        print('\n------ Repeat %3d ------' % (i + 1))
        for cat in sorted(shape_ious.keys()):
            print('%s: %0.6f'%(cat, shape_ious[cat]))
        print('Class_mIoU: %0.6f' % (mean_class_ious))
        print('Instance_mIoU: %0.6f' % (np.mean(instance_ious)))

        if mean_class_ious > global_Class_mIoU:
            global_Class_mIoU = mean_class_ious
            global_Inst_mIoU = np.mean(instance_ious)
                
    print('\nBest voting Class_mIoU = %0.6f, Instance_mIoU = %0.6f' % (global_Class_mIoU, global_Inst_mIoU))
        
if __name__ == '__main__':
    main()