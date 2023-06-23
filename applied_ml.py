#!/usr/bin/env python
# coding: utf-8

# In[12]:


## Importing Libraries
# General
import datetime
from time import time
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

# Visualization
import matplotlib.pyplot as plt

import seaborn as sns

# OpenCV
import cv2

# ScikitLearn for Data Splitting
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold

# Pytorch
import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import nms
from torchvision import datasets, transforms

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler


# In[13]:




DAY_TRAIN_PATH = '/home/tosy/Desktop/Project Applied ML/Annotations/Annotations/dayTrain/'
NIGHT_TRAIN_PATH = '/home/tosy/Desktop/Project Applied ML/Annotations/Annotations/nightTrain/'


# In[14]:



train_day = []
for clipName in (sorted(os.listdir(DAY_TRAIN_PATH))):
    if 'dayClip' not in clipName:
        continue
    df = pd.read_csv(os.path.join(DAY_TRAIN_PATH,clipName,'frameAnnotationsBOX.csv'),sep=';')
    train_day.append(df)
    
train_day_df = pd.concat(train_day,axis=0)
train_day_df['isNight'] = 0
    
train_night = []
for clipName in (sorted(os.listdir(NIGHT_TRAIN_PATH))):
    if 'nightClip' not in clipName:
        continue
    df = pd.read_csv(os.path.join(NIGHT_TRAIN_PATH,clipName,'frameAnnotationsBOX.csv'),sep=';')
    train_night.append(df)

train_night_df = pd.concat(train_night,axis=0)
train_night_df['isNight'] = 1

df = pd.concat([train_day_df,train_night_df],axis=0)


# In[15]:




# Droppin duplicate columns & "Origin file" as we don't need it
df = df.drop(['Origin file','Origin track','Origin track frame number'],axis=1)


# In[16]:



def changeFilename(x):
    filename = x.Filename
    isNight = x.isNight
    
    splitted = filename.split('/')
    clipName = splitted[-1].split('--')[0]
    if isNight:
        return os.path.join(f'nightTrain/nightTrain/{clipName}/frames/{splitted[-1]}')
    else:
        return os.path.join(f'dayTrain/dayTrain/{clipName}/frames/{splitted[-1]}')

df['Filename'] = df.apply(changeFilename,axis=1)


# In[17]:


df


# In[18]:


label_convertin = {'go':1,'warning':2, 'stop' :3}
def changeAnnotation(x):
    if 'go' in x['Annotation tag']:
        return label_convertin['go']
    elif 'warning' in x['Annotation tag']:
        return label_convertin['warning']
    elif 'stop' in x['Annotation tag']:
        return label_convertin['stop']
    
df['Annotation tag'] = df.apply(changeAnnotation,axis=1)


# In[19]:


df.columns = ['Filename','label','x_min','y_min','x_max','y_max','number','isNight']


# In[20]:


df


# In[10]:


import os
import shutil
names = df['Filename'].tolist()
label = df['isNight'].tolist()
for i in range(len(names)):
    if label[i] == 1:
        shutil.copy('./'+names[i], './data/night')
    else:
        shutil.copy('./'+names[i], './data/day')


# ### CLASSIFICATION TASK

# In[10]:


####Doing Classification
import PIL
df1 = df[['Filename', 'isNight']]
df1


# In[11]:


df1.drop_duplicates(subset = 'Filename', keep = 'first', inplace = True)
df1.reset_index(inplace= True)
df1.drop(['index'],axis= 1, inplace = True)


# In[12]:


df_train, df_test = train_test_split(df1, test_size=0.2, random_state=42, stratify = df1['isNight'])


# In[13]:


df_train


# In[14]:


import os
import shutil
names = df_train['Filename'].tolist()
label = df_train['isNight'].tolist()
for i in range(len(names)):
    if label[i] == 0:
        s = names[i].split('/')[-1]
        shutil.copy(f'./data/day/{s}', './data/train')
    elif label[i] == 1: 
        s = names[i].split('/')[-1]
        shutil.copy(f'./data/night/{s}', './data/train')


# In[15]:


import os
import shutil
names = df_test['Filename'].tolist()
label = df_test['isNight'].tolist()
for i in range(len(names)):
    if label[i] == 0:
        s = names[i].split('/')[-1]
        shutil.copy(f'./data/day/{s}', './data/test')
    elif label[i] == 1: 
        s = names[i].split('/')[-1]
        shutil.copy(f'./data/night/{s}', './data/test')


# In[16]:


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, images_folder, transform = None):
        self.df = df
        self.images_folder = images_folder
        self.transform = transforms.Compose([transforms.Resize((32,32)),
                                transforms.ToTensor()])


    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        filename = self.df.iloc[index, 0].split('/')[-1]
        label = self.df.iloc[index, 1]
        image =  PIL.Image.open(os.path.join(self.images_folder, filename))
        image = self.transform(image)
        return image, label


# In[17]:


train_dataset = CustomDataset(df_train, '/home/tosy/Desktop/Project Applied ML/data/train' )
val_dataset = CustomDataset(df_test, '/home/tosy/Desktop/Project Applied ML/data/test'  )


# In[18]:


train_dataset[0]


# In[19]:


## DATA LOADER
BATCH_SIZE = 16
train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader =  DataLoader(dataset = val_dataset, batch_size = BATCH_SIZE, shuffle = False)


# ### Modelling for Classification
# 

# In[20]:


import torch.nn as nn
LEARNING_RATE = 0.001
N_EPOCHS = 10
IMG_SIZE = 32
N_CLASSES = 2


# In[21]:


class LeNet5(nn.Module):
    def __init__(self,n_classes):
        super(LeNet5, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5, stride = 1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size = 2,stride = 2),
            nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, stride = 1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size = 2,stride = 2),
            nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = 5, stride = 1),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features = 120, out_features =84),
            nn.Tanh(),
            nn.Linear(in_features = 84, out_features = n_classes)
        )
    def forward(self,x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# In[22]:


def train(train_loader, model, criterion, optimizer,device):
    model.train()

    train_loss = 0
    correct = 0

    for X,y_true in train_loader:
        optimizer.zero_grad()
        X = X.to(device)
        #y_true= y_true.unsqueeze(-1)
        y_true = y_true.to(device)
        y_hat = model(X)

        loss = criterion(y_hat,y_true)
        train_loss += loss.item()

        pred = y_hat.argmax(dim = 1, keepdim = True)
        correct += pred.eq(y_true.view_as(pred)).sum().item()

        loss.backward()
        optimizer.step()

    epoch_loss = train_loss / len(train_loader.dataset)
    acc = correct / len(train_loader.dataset)

    return model, optimizer, epoch_loss, acc


# In[23]:


def test(test_loader, model, criterion,device):
    model.eval()

    test_loss = 0
    correct = 0

    for X,y_true in test_loader:
        X = X.to(device)
       # y_true= y_true.unsqueeze(-1)
        y_true = y_true.to(device)

        y_hat = model(X)
        loss = criterion(y_hat,y_true)
        test_loss += loss.item()


        pred = y_hat.argmax(dim = 1, keepdim = True)
        correct += pred.eq(y_true.view_as(pred)).sum().item()

    epoch_loss = test_loss / len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)

    return model, epoch_loss, acc


# In[24]:


def trainig_loop(model, criterion, optimizer, train_loader, test_loader, epochs, device, print_every = 1):

    train_losses = []
    test_losses = []
    for epoch in range(epochs):

        model, optimizer, train_loss, train_acc = train(train_loader, model,criterion,optimizer,device)
        train_losses.append(train_loss)

        with torch.no_grad():
            model, test_loss,test_acc = test(test_loader, model, criterion,device)
            test_losses.append(test_loss)

        if epoch % print_every == (print_every - 1):

            print(f'Epoch: {epoch}\t',
                  f'Train Loss: {train_loss:.4f}\t',
                  f'Test Loss: {test_loss:.4f}\t',
                  f'Train accuracy: {100*train_acc:.2f}\t',
                  f'Test accuracy: {100*test_acc:.2f}\t'
                  )
    return model,optimizer, (train_losses, test_losses)


# In[28]:



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model =  LeNet5(N_CLASSES).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

model, optimizer, _ =  trainig_loop(model, criterion, optimizer, train_loader, test_loader, N_EPOCHS,DEVICE)


# In[22]:


classes = {0:'day',1:'night'}


# In[30]:


import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(test_loader)
images, labels = next(dataiter)
labels1 = labels.tolist()

# show images
imshow(torchvision.utils.make_grid(images))
#print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(BATCH_SIZE)))
# print labels
print(' '.join(f'{classes[labels1[j]]}' for j in range(BATCH_SIZE)))
##### IMPORTANT
images, labels = images.cuda(), labels.cuda()


# In[31]:


outputs = model(images)

_, predicted = torch.max(outputs, 1)
predicted = predicted.tolist()

print('Predicted: ', ' '.join(f'{classes[predicted[j]]}'
                              for j in range(16)))


# ### Object detection model

# In[10]:


df


# In[11]:


#importing the Image class from PIL package
from PIL import Image
#read the image, creating an object
im = Image.open(df.iloc[0,0])
#show picture
im.show()


# In[12]:


df


# In[23]:


annotation_tags = df['label'].unique()


# In[24]:


idx_to_label = {1:'go', 2: 'warning', 3: 'stop'}


# In[25]:


fig, ax = plt.subplots(len(annotation_tags),1,figsize=(15,10*len(annotation_tags)))
for i, tag in enumerate(annotation_tags):
    sample = df[df['label']==tag].sample(1)
    bbox = sample[['x_min','y_min','x_max','y_max']].values[0]

    image = cv2.imread(sample['Filename'].values[0])
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    flipped = cv2.flip(image,1)
    

    cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]), (220,0,0),2)

    ax[i].set_title(idx_to_label[tag])
    ax[i].set_axis_off()
    ax[i].imshow(image)
    

    


# In[26]:


import albumentations as A
from albumentations.pytorch import ToTensorV2


# In[27]:


df['clipNames'] = df[['Filename']].applymap(lambda x: x.split('/')[2])
df['clipNames'].unique()


# In[28]:


df


# In[29]:




def split(df,p=0.25):
    clipNames = sorted(df['clipNames'].unique())

    nightClips = [name for name in clipNames if 'night' in name]
    dayClips = [name for name in clipNames if 'day' in name]

    testNightClipNames = list(np.random.choice(nightClips,int(len(nightClips)*p)))
    testDayClipNames = list(np.random.choice(dayClips,int(len(dayClips)*p)))
    testClipNames = testNightClipNames + testDayClipNames

    trainDayClipNames = list(set(dayClips) - set(testDayClipNames))
    trainNightClipNames = list(set(nightClips) - set(testNightClipNames))
    trainClipNames = trainNightClipNames + trainDayClipNames
    
    train_df = df[df.clipNames.isin(trainClipNames)]
    test_df = df[df.clipNames.isin(testClipNames)]
    
    return train_df, test_df



# In[39]:


df_train, df_test = split(df)


# In[93]:


import os
import shutil
names = df_train.drop_duplicates(['Filename'])['Filename'].tolist()
label = df_train.drop_duplicates(['Filename'])['isNight'].tolist()
for i in range(len(names)):
    if label[i] == 0:
        s = names[i].split('/')[-1]
        shutil.copy(f'./data/day/{s}', './data/train1')
    elif label[i] == 1: 
        s = names[i].split('/')[-1]
        shutil.copy(f'./data/night/{s}', './data/train1')


# In[94]:


import os
import shutil
names =  df_test.drop_duplicates(['Filename'])['Filename'].tolist()
label = df_test.drop_duplicates(['Filename'])['isNight'].tolist()
for i in range(len(names)):
    if label[i] == 0:
        s = names[i].split('/')[-1]
        shutil.copy(f'./data/day/{s}', './data/test1')
    elif label[i] == 1: 
        s = names[i].split('/')[-1]
        shutil.copy(f'./data/night/{s}', './data/test1')


# In[40]:



EPOCHS = 4
BATCH_SIZE = 16


# In[41]:


df_train


# In[42]:




class TrafficLightsDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms=None):
        super().__init__()

        # Image_ids will be the "Filename" here
        self.image_ids = df.Filename.unique()
        self.df = df
        self.transforms = transforms
        
    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df.Filename == image_id]

        # Reading Image
        image = cv2.imread(image_id)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        
        # Bounding Boxes
        boxes = records[['x_min','y_min','x_max','y_max']].values
        boxes = torch.as_tensor(boxes,dtype=torch.float32)
        
        # Area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # Labels of the object detected
        labels = torch.as_tensor(records.label.values, dtype=torch.int64)
        
        iscrowd = torch.zeros_like(labels, dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            
            # target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
            target['boxes'] = torch.as_tensor(sample['bboxes'],dtype=torch.float32)
            target['labels'] = torch.as_tensor(sample['labels'])
            
        return image, target, image_id


# In[43]:



def collate_fn(batch):
    return tuple(zip(*batch))


# In[44]:



def getTrainTransform():
    return A.Compose([
        A.Resize(height=512, width=512, p=1),
        A.augmentations.geometric.transforms.Flip(0.5),
        A.augmentations.geometric.rotate.RandomRotate90(p = 0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})



# For Test Data
def getTestTransform():
    return A.Compose([
        A.Resize(height=512, width=512, p=1),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


# In[45]:




trainDataset = TrafficLightsDataset(df_train,getTrainTransform())
testDataset = TrafficLightsDataset(df_test,getTestTransform())


# In[46]:


trainDataLoader = DataLoader(
    trainDataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn = collate_fn)


testDataLoader = DataLoader(
    testDataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn= collate_fn)


# In[47]:



images, targets, image_ids = next(iter(trainDataLoader))

boxes = targets[0]['boxes'].cpu().numpy().astype(np.int32)
image = images[0].permute(1,2,0).cpu().numpy()
def displayImage(image, boxes):
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for box in boxes:
        cv2.rectangle(image,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (220, 0, 0), 3)

    ax.set_axis_off()
    ax.imshow(image)

    plt.show()


# In[48]:


displayImage(image,boxes)


# In[49]:


from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
def get_object_detection_model(num_classes = 4, 
                               feature_extraction = True):
    """
    Inputs
        num_classes: int
            Number of classes to predict. Must include the 
            background which is class 0 by definition!
        feature_extraction: bool
            Flag indicating whether to freeze the pre-trained 
            weights. If set to True the pre-trained weights will be  
            frozen and not be updated during.
    Returns
        model: FasterRCNN
    """
    # Load the pretrained faster r-cnn model.
    model = fasterrcnn_resnet50_fpn(pretrained = True)   
    if feature_extraction == True:
        ct =0 
        for child in model.children():
            ct += 1
            if ct < 7:
                for param in child.parameters():
                    param.requires_grad = False
            # Replace the original 91 class top layer with a new layer
    # tailored for num_classes.
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats,
                                                   num_classes)   
    return model


# In[50]:


model = get_object_detection_model(feature_extraction = True)


# In[51]:


for k,v in model.backbone.body.named_parameters():
    print(k)


# In[52]:




DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(DEVICE)

params = [p for p in model.parameters() if p.requires_grad]
# Optimizers
optimizer = torch.optim.Adam(params)

# LR Scheduler
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)


# In[53]:




# Average loss -> (Total-Loss / Total-Iterations)
class LossAverager:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[54]:




lossHist = LossAverager()
valLossHist = LossAverager()

for epoch in range(EPOCHS):
    
    start_time = time()
    model.train()
    lossHist.reset()
    
    for images, targets, image_ids in tqdm(trainDataLoader):
        
        images = torch.stack(images).to(DEVICE)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        bs = images.shape[0]
        
        loss_dict = model(images, targets)
        
        totalLoss = sum(loss for loss in loss_dict.values())
        lossValue = totalLoss.item()
        
        lossHist.update(lossValue,bs)

        optimizer.zero_grad()
        totalLoss.backward()
        optimizer.step()
    
    # LR Update
    if lr_scheduler is not None:
        lr_scheduler.step(totalLoss)

    print(f"[{str(datetime.timedelta(seconds = time() - start_time))[2:7]}]")
    print(f"Epoch {epoch}/{EPOCHS}")
    print(f"Train loss: {lossHist.avg}")


# In[55]:




model.eval()
images, targets, image_ids = next(iter(testDataLoader))
images = torch.stack(images).to(DEVICE)

outputs = model(images)


# In[56]:




def filterBoxes(output,nms_th=0.3,score_threshold=0.5):
    
    boxes = output['boxes']
    scores = output['scores']
    labels = output['labels']
    
    # Non Max Supression
    mask = nms(boxes,scores,nms_th)
    
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    boxes = boxes.data.cpu().numpy().astype(np.int32)
    scores = scores.data.cpu().numpy()
    labels = labels.data.cpu().numpy()
    
    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    return boxes, scores, labels


# In[57]:




def displayPredictions(image_id,output,nms_th=0.3,score_threshold=0.5):
    
    boxes,scores,labels = filterBoxes(output,nms_th,score_threshold)
    
    # Preprocessing
    image = cv2.imread(image_id)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = cv2.resize(image,(512,512))
    image /= 255.0
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    colors = {1:(0,255,0), 2:(255,255,0), 3:(255,0,0)}
    
    for box,label in zip(boxes,labels):
        image = cv2.rectangle(image,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      colors[label], 2)

    ax.set_axis_off()
    ax.imshow(image)

    plt.show()


# In[209]:


image_ids[9]


# In[124]:


dataiter = iter(testDataLoader)
images, targets, image_ids= next(dataiter)
images = torch.stack(images).to(DEVICE)

outputs = model(images)
displayPredictions(image_ids[1],outputs[1],0.2,0.4)


# In[119]:



dataiter = iter(testDataLoader)
images, targets, image_ids= next(dataiter)
images = torch.stack(images).to(DEVICE)

outputs = model(images)
displayPredictions(image_ids[9],outputs[9],0.2,0.4)


# In[125]:


#### YOLOV8
import cv2
import numpy as np
import os
import csv


# In[130]:


df_train


# In[134]:



def pascal_voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
    return [((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h]


# In[140]:


import os
import shutil
names = df_test.drop_duplicates(['Filename'])['Filename'].tolist()
label = df_test.drop_duplicates(['Filename'])['isNight'].tolist()
for i in range(len(names)):
    s = names[i].split('/')[-1]
    shutil.copy(f'./data/test1/{s}', './data/images')


# In[159]:


df_train['Name'] = df_train[['Filename']].applymap(lambda x: x.split('/')[-1])


# In[173]:


df_test['Name'] = df_test[['Filename']].applymap(lambda x: x.split('/')[-1])


# In[192]:


folder_path = 'data/train1/images'

txt_folder_path = 'data/train1' 

#check class labels from the csv file
def pascal_voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
    return [((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h]

s = 0
for filename in os.listdir(folder_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        image = cv2.imread(os.path.join(folder_path, filename))
        height, width, _ = image.shape

        output_txt_folder = os.path.join(txt_folder_path, 'labels')
        os.makedirs(output_txt_folder, exist_ok=True)
        output_txt_path = os.path.join(output_txt_folder, f'{os.path.splitext(filename)[0]}.txt')
        n = 0
        with open(output_txt_path, 'w') as file:
            for row in df_train[df_train['Name'] == filename].values:
                label = row[1]-1
                x,y,w,h = pascal_voc_to_yolo(row[2], row[3], row[4], row[5], width, height)
                if n ==0:
                    file.write(f"{label} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                else:
                    file.write(f"\n{label} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                n += 1


# In[193]:


folder_path = 'data/test1/images'

txt_folder_path = 'data/test1' 

#check class labels from the csv file
def pascal_voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
    return [((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h]

s = 0
for filename in os.listdir(folder_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        image = cv2.imread(os.path.join(folder_path, filename))
        height, width, _ = image.shape

        output_txt_folder = os.path.join(txt_folder_path, 'labels')
        os.makedirs(output_txt_folder, exist_ok=True)
        output_txt_path = os.path.join(output_txt_folder, f'{os.path.splitext(filename)[0]}.txt')
        n = 0
        with open(output_txt_path, 'w') as file:
            for row in df_test[df_test['Name'] == filename].values:
                label = row[1]-1
                x,y,w,h = pascal_voc_to_yolo(row[2], row[3], row[4], row[5], width, height)
                if n ==0:
                    file.write(f"{label} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                else:
                    file.write(f"\n{label} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                n += 1


# In[208]:


from ultralytics import YOLO
path = '/home/tosy/Desktop/yolov8/runs/detect/train5/weights/best.pt'
model = YOLO(path)
model.predict('/home/tosy/Desktop/Project Applied ML/data/test1/images/dayClip1--00011.jpg',device = 'cpu',save = True)


# In[211]:



model.predict('/home/tosy/Desktop/Project Applied ML/data/test1/images',device = 'cpu',save = True)


# In[ ]:




