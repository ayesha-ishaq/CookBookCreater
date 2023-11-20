import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blip import blip_decoder
import utils
from data import create_dataset, create_sampler, create_loader
from data.utils import save_result, save_ann, custom_title_eval


@torch.no_grad()
def evaluate(model, data_loader, device, config):
    # evaluate
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Caption generation:'
    print_freq = 10

    result = []
    gt = {'annotations':[], 'images':[]}
    for image, captions_gt, image_id in metric_logger.log_every(data_loader, print_freq, header): 
        
        image = image.to(device)       
        
        captions = model.generate(image, sample=True, num_beams=config['top_p'], max_length=config['max_length'], 
                                  min_length=config['min_length'])
       
        for caption, img_id, caption_gt in zip(captions, image_id, captions_gt):
            result.append({"image_id": img_id.item(), "caption": caption})
            gt['annotations'].append({"image_id": img_id.item(), "caption": caption_gt, 'id': img_id.item()})
            gt['images'].append({"id": img_id.item()})
  
    return result, gt


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating captioning dataset")
    _, val_dataset = create_dataset('custom', config)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([val_dataset], [False], num_tasks, global_rank)         
    else:
        samplers = [None]
    
    val_loader = create_loader([val_dataset],samplers, 
                 batch_size=[config['batch_size']]*1, num_workers=[4],
                 is_trains=[False], collate_fns=[None])[0]        

    #### Model #### 
    print("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                           prompt=config['prompt'])

    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    val_result, gt = evaluate(model_without_ddp, val_loader, device, config)  
    val_result_file = save_result(val_result, args.result_dir, 'val_result', remove_duplicate='image_id')
    val_gt_file = save_ann(gt, args.result_dir, 'val_gt') 

    if utils.is_main_process(): 
        eval_val = custom_title_eval(val_gt_file, val_result_file,'val')
   
        
        log_stats = {**{f'val_{k}': v for k, v in eval_val.eval.items()}}
        with open(os.path.join(args.output_dir, "log_eval_base.txt"),"a") as f:
            f.write(json.dumps(log_stats) + "\n")   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/custom_recipe1m.yaml')
    parser.add_argument('--output_dir', default='./output/evalution_val')        
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)