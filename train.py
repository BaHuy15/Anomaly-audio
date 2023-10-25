
import torch
import glob
import torch.nn as nn
from model import AE
import numpy as np
import random
from dataset import ASDDataset
from collections import defaultdict
from datetime import timedelta
import time
class args:
    frames=5
    n_mels=64
    frames=5
    n_fft=1024
    hop_length=512
    lr=1e-02
    w_d = 1e-3 
    epochs =80
    seed=42
    batch_size=128
    file_path=glob.glob('/home/tonyhuy/my_project/audio_classification/content/DATASET_FINAL/OK/train/*.wav')

#================================== Config model devices,optimizer=========================================#
seed = args.seed 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
metrics = defaultdict(list)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AE()
model.to(device)
criterion = nn.MSELoss(reduction='mean')
# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.w_d)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,weight_decay=args.w_d)

def save_checkpoint(epoch, model, optimizer, filename):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)
#=================================== Prepare training data==================================================#
train_set = ASDDataset(args)
train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=20,
            pin_memory=True,
            drop_last=True)
model.train()
start = time.time()
for epoch in range(args.epochs):
    ep_start = time.time()
    running_loss = 0.0
    for bx, (data) in enumerate(train_loader):
        sample = model(data.to(device))
        # print(f'data:{data} and \n sample: {sample}')
        loss = criterion(data.to(device), sample)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss/len(train_set)
    metrics['train_loss'].append(epoch_loss)
    filename=f'/home/tonyhuy/my_project/anomaly_audio/checkpoint/epoch_{epoch}.pth'
    save_checkpoint(epoch, model, optimizer, filename)
    ep_end = time.time()
    print('-----------------------------------------------')
    print('[EPOCH] {}/{}\n[LOSS] {}'.format(epoch+1,args.epochs,epoch_loss))
    print('Epoch Complete in {}'.format(timedelta(seconds=ep_end-ep_start)))
end = time.time()
print('-----------------------------------------------')
print('[System Complete: {}]'.format(timedelta(seconds=end-start)))
