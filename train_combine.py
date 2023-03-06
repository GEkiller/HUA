import os
import math
import argparse

import sys
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
# from models.volo import *
from My_model import *
from densenet import *
from Feature_Combine import *
from PIL import Image
from tqdm import tqdm

from my_utils import train_one_epoch, evaluate, read_csv, count_mean_std
from utils import load_pretrained_weights

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# The NO of dataset
dataset_num = 5

dense_base = '../liver-rgb/Good-model/CurrentDataset/'
volo_base = './Good-model/应该是最终的数据集/volo/'

dense_base = dense_base + str(dataset_num) + '/'
volo_base = volo_base + str(dataset_num) + '/'

dense_path = [dense_base + i for i in os.listdir(dense_base)]
volo_path = [volo_base + i for i in os.listdir(volo_base)]


train_images_path, train_images_label = read_csv('/media/hsy/1882C80C82C7EC76/ZJH/Cross-Validation-Dataset'
                                                 '/Current_dataset/' + str(dataset_num) + '/train.csv')

train_transformer = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.85, 0.53, 0.669], [0.074, 0.17, 0.13])
])
train_dataset = MyDataSet(images_path=train_images_path,
                          images_class=train_images_label,
                          transform=train_transformer)

batch_size = 8
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
print('Using {} dataloader workers every process'.format(nw))
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=nw,
                                           collate_fn=train_dataset.collate_fn)

model_volo = volo_d3(num_classes=3).to(device)
model_dense = densenet121(mode='rgb', num_classes=3).to(device)

model = Final_block().to(device)

# pre_w = torch.load('../liver-rgb/Good-model/CurrentDataset/2/model-271.pth', map_location=device)
# pre_w = torch.load('../liver-rgb/Good-model/CurrentDataset/1/model-264.pth', map_location=device)
pre_w = torch.load(dense_path[0], map_location=device)

# fc_features = model_dense.classifier.in_features
# model_dense.classifier = nn.Linear(fc_features, 3)
model_dense.load_state_dict(pre_w, strict=False)
model_dense.eval()

# model_weight_path = "./Good-model/应该是最终的数据集/volo/2/model-267.pth"
# model_weight_path = "./Good-model/应该是最终的数据集/volo/1/model-263.pth"
# model_weight_path = "./Good-model/应该是最终的数据集/volo/3/model-277.pth"
model_volo.load_state_dict(torch.load(volo_path[0], map_location=device))
model_volo.eval()

# Parameter definition
lr = 1e-5
lrf = 0.01
epochs = 100

pg = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(pg, lr=lr)

# Scheduler https://arxiv.org/pdf/1812.01187.pdf
lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
# lf = lambda x: 0.95 ** x + lrf
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
for epoch in range(epochs):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    # loss_function = FocalLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(train_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        feature_volo = model_volo(images.to(device))[1]
        feature_dense = model_dense(images.to(device))[1]
        input = torch.cat((feature_volo, feature_dense), 1)
        pred = model(input)
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    if epoch >= 50:
        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

# for file in train_images_path:
# load image
# img_path = "/media/hsy/1882C80C82C7EC76/Level-patch-1024-patient/T-3/TMA2-F11_RGB_T16/TMA2-F11_RGB_T16-13.jpg"
# assert os.path.exists(file), "file: '{}' dose not exist.".format(file)
# img = Image.open(file)
# plt.imshow(img)
# [N, C, H, W]
# img = train_transformer(img)
# expand batch dimension
# img = torch.unsqueeze(img, dim=0)
