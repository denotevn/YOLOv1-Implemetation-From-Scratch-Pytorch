import torch
import torch.utils.data as data
from PIL import Image
import os
import numpy as np

class YOLODataset(data.Dataset):
    def __init__(self, image_dir, label_dir, S=7, B=2, C=20, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        
        self.S = S
        self.B = B
        self.C = C
        
        self.label_files = os.listdir(label_dir)
        self.image_files = [os.path.join(image_dir, f.split('.')[0] + '.jpg') for f in self.label_files]
        
    def __len__(self):
        return len(self.label_files)
    
    def __getitem__(self, index):
        label_file = self.label_files[index]
        image_file = self.image_files[index]
        
        label_path = os.path.join(self.label_dir, label_file)
        image_path = os.path.join(self.image_dir, label_file.split('.')[0] + '.jpg')
        
        image = Image.open(image_path).convert('RGB')
        label = np.loadtxt(label_path).reshape(-1, 5)
        
        boxes = label[:, 1:]
        boxes[:, 0] *= image.width
        boxes[:, 1] *= image.height
        boxes[:, 2] *= image.width
        boxes[:, 3] *= image.height
        
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        
        target = torch.zeros((self.S, self.S, self.B * 5 + self.C))
        
        for box in boxes:
            x, y, w, h = box
            xc = x / (image.width / self.S)
            yc = y / (image.height / self.S)
            
            wi = w / image.width
            hi = h / image.height
            
            i, j = int(xc), int(yc)
            
            anchor_ious = []
            for k in range(self.B):
                anchor = self.anchors[k]
                iou = self.box_iou(anchor, [wi, hi])
                anchor_ious.append(iou)
                
            best_anchor = int(np.argmax(anchor_ious))
            
            target[i, j, best_anchor * 5] = xc - i
            target[i, j, best_anchor * 5 + 1] = yc - j
            target[i, j, best_anchor * 5 + 2] = np.log(wi / self.anchors[best_anchor][0] + 1e-16)
            target[i, j, best_anchor * 5 + 3] = np.log(hi / self.anchors[best_anchor][1] + 1e-16)
            target[i, j, best_anchor * 5 + 4] = 1
            
            target[i, j, self.B * 5 + int(label[i, 0])] = 1
            
        if self.transform:
            image = self.transform(image)
        
        return image, target
