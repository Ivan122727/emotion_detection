# Импорт библиотек
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
 
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import models
 
from torch.utils.data import DataLoader
from torchvision.transforms import v2
 
import os
from glob import glob
from tqdm import tqdm
 
 
# Создание набора данных
class MyDataset(torch.utils.data.Dataset):
    def __init__(self , dataframe , transforms_):
        self.df = dataframe
        self.transforms_ = transforms_
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self ,index):
        img_path = self.df.iloc[index]['path']
        img = Image.open(img_path).convert("RGB")
        transformed_img = self.transforms_(img)
        class_id = self.df.iloc[index]['class_id']
        return transformed_img , class_id
 
 
# Определение функций тренировки и тестирования
#  функция тренировки, которая обучает модель на одной эпохе
def train(device, dataloader , model , loss_fn , optimizer , lr_scheduler):
    size = 0
    num_batches = len(dataloader)
 
    model.train()
    epoch_loss , epoch_correct = 0 , 0
 
    for i ,(data_ , target_) in enumerate(dataloader):
        target_ = target_.type(torch.LongTensor)
        data_ , target_ = data_.to(device) , target_.to(device)
 
        outputs = model(data_)
 
        loss = loss_fn(outputs , target_)
        epoch_loss =+ loss.item()
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        _ , pred = torch.max(outputs , dim = 1)
        epoch_correct = epoch_correct + torch.sum(pred == target_).item()
        size += target_.shape[0]
    lr_scheduler.step()
    return epoch_correct/size , epoch_loss / num_batches
 
 
# функция тестирования, которая оценивает модель на валидационных данных.
def test(device, dataloader , model , loss_fn):
    size = 0
    num_baches = len(dataloader)
    epoch_loss , epoch_correct= 0 ,0
    with torch.no_grad():
        model.eval()
        for i, (data_ , target_) in enumerate(dataloader):
            target_ = target_.type(torch.LongTensor)
            data_ , target_ = data_.to(device) , target_.to(device)
 
            outputs = model(data_)
 
            loss = loss_fn(outputs , target_)
 
            epoch_loss = epoch_loss + loss.item()
            _,pred = torch.max(outputs , dim = 1)
            epoch_correct += torch.sum(pred == target_).item()
 
            size+= target_.shape[0]
    return epoch_correct/size  , epoch_loss / num_baches
 
 
if __name__ == '__main__':
    # Подготовка данных
    train_df = pd.DataFrame({"path":[] , "label":[] , "class_id":[]})
    train_path = os.path.join(os.getcwd(), 'data', 'train')
    label_list = ['angry','disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    img_list =  glob(os.path.join(train_path, '**', '*.png'), recursive=True)
 
 
    print(f"Found {len(img_list)} training images.")
 
 
    for img in img_list:
        file_name = os.path.splitext(img)[0].split("\\")[-1]
        label_name = os.path.splitext(img)[0].split("\\")[-2]
        new_data = pd.DataFrame({"path":img , "label": label_name , "class_id": label_list.index(label_name)} , index=[1])
        train_df = pd.concat([train_df , new_data] , ignore_index = True)
 
 
    if train_df.empty:
        print("Training DataFrame is empty!")
    else:
        print(f"Training DataFrame has {len(train_df)} entries.")
 
 
    train_df[["path"]] = train_df[["path"]].astype(str)
    train_df[["label"]] = train_df[["label"]].astype(str)
    train_df[["class_id"]] = train_df[["class_id"]].astype(int)
 
 
    # Создание DataFrame для валидационных данных
    val_df = pd.DataFrame({"path":[] , "label":[] , "class_id":[]})
    val_path = os.path.join(os.getcwd(), 'data', 'test')
    img_list = glob(os.path.join(val_path, '**', '*.png'), recursive=True)
 
 
    print(f"Found {len(img_list)} validation images.")
 
 
    for img in img_list:
        file_name = os.path.splitext(img)[0].split("\\")[-1]
        label_name = os.path.splitext(img)[0].split("\\")[-2]
        new_data = pd.DataFrame({"path": img , "label": label_name , "class_id": label_list.index(label_name)} , index=[1])
        val_df = pd.concat([val_df , new_data]  , ignore_index=True)
 
 
    if train_df.empty:
        print("Validation DataFrame is empty!")
    else:
        print(f"Validation DataFrame has {len(val_df)} entries.")
 
 
    val_df[['path']] = val_df[['path']].astype(str)
    val_df[['label']] = val_df[['label']].astype(str)
    val_df[['class_id']] = val_df[['class_id']].astype(int)
 
 
    # Преобразования изображений
    train_transforms = v2.Compose([
        v2.Resize(265),
        v2.RandomResizedCrop(size = (224 , 224) , antialias = True),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(.5),
        v2.RandomAffine(degrees=(-10,10),translate=(.1,.1), scale=(.9,1.1)),
        v2.RandomErasing(p=.5,scale = (.1,.15)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean = [.485,.456,.406] , std = [0.229 , 0.224 , 0.225])
    ])
 
 
    test_transforms = v2.Compose([
        v2.Resize((224,224)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean = [.485,.456,.406] , std = [0.229 , 0.224 , 0.225])
    ])
 
 
    # Создание DataLoader
    BATCH_SIZE = 6 # 6
    num_workers = 6 # Кол-во ядер у процессора
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
 
    train_dataset = MyDataset(train_df , train_transforms)
    val_dataset = MyDataset(val_df , test_transforms)
 
 
    # Длины набора данных
    print(f"Length of train_dataset: {len(train_dataset)=}")
    print(f"Length of val_dataset: {len(val_dataset)=}")
 
    train_loader = DataLoader(train_dataset , batch_size=BATCH_SIZE , shuffle = True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset , batch_size=BATCH_SIZE, num_workers=num_workers)
 
 
    # Проверка длины DataLoader
    print(f"Length of train_loader: {len(train_loader)=}")
    print(f"Length of val_loader: {len(val_loader)=}")
 
 
    # Создание модели
    class_size = len(label_list)
    model = models.swin_v2_b(weights= 'DEFAULT')
 
 
    model.head = nn.Linear(in_features = model.head.in_features,
                        out_features = class_size)
    
 
    # Обучение модели
    EPOCHS = 50
    logs = {"train_loss":[] , "train_acc":[] , "val_loss":[] , "val_acc":[]}
 
 
    if os.path.exists('checkpoints') == False:
        os.mkdir('checkpoints')
 
 
    criterion = nn.CrossEntropyLoss()
 
 
    learning_rate = 0.0001
    momentum = .9
    weight_decay = .1
 
 
    optmizer = torch.optim.AdamW(model.parameters() , lr = learning_rate)
 
 
    lr_milestones = [7 , 14, 21 , 28 , 35]
    multi_step_lr_scheduler = lr_scheduler.MultiStepLR(optmizer ,
                                                       milestones=lr_milestones,
                                                       gamma = .1)
 
 
    # Параметры ранней остановки
    patience = 8 # Если 8 эпох нету улучшений
    counter = 0 # Счетчик эпох без улучшений
    best_loss = np.inf # Начальное значение лучшей валидационной потери
 
 
    model.to(device)
 
 
    for epoch in tqdm(range(EPOCHS)):
        train_acc , train_loss = train(device, train_loader ,
                                       model ,
                                       criterion ,
                                       optmizer ,
                                       multi_step_lr_scheduler)
        val_acc , val_loss = test(device, val_loader , model , criterion)
        print(f'epoch:{epoch} \
        train_loss = {train_loss:.4f} , train_acc:{train_acc:.4f} \
        val_loss = {val_loss:.4f} , val_acc:{val_acc:.4f} \
        learning rate: {optmizer.param_groups[0]["lr"]}')
        logs['train_loss'].append(train_loss)
        logs['train_acc'].append(train_acc)
        logs['val_loss'].append(val_loss)
        logs['val_acc'].append(val_acc)
 
        if val_loss < best_loss:
            counter = 0
            best_loss = val_loss
            torch.save(model.state_dict() , "checkpoints\\best.pth")
        else:
            counter+=1
        if counter >= patience:
            print("Early stop !")
            break
 
    #  Визуализация результатов
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(logs['train_loss'],label='Train_Loss')
    plt.plot(logs['val_loss'],label='Validation_Loss')
    plt.title('Train_Loss & Validation_Loss',fontsize=20)
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(logs['train_acc'],label='Train_Accuracy')
    plt.plot(logs['val_acc'],label='Validation_Accuracy')
    plt.title('Train_Accuracy & Validation_Accuracy',fontsize=20)
    plt.legend()
    plt.show()