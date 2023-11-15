import os
import json

from torch.utils.data import Dataset

from PIL import Image

from data.utils import pre_caption

class CustomDataset(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt='', train=True):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        if train:
            filename = 'train.json'
        else:
            filename = 'val.json'

        self.annotation = json.load(open(os.path.join(ann_root,filename),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image'].split('/')[-1].strip('.')[0]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root, ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 
        img_id = ann['image'].split('/')[-1].strip('.')[0]

        return image, caption, self.img_ids[img_id] 