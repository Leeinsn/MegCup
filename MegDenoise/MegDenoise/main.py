#!/usr/bin/env python
# coding: utf-8

# In[1]:


import megengine as mge
import megengine.module as M
import megengine.functional as F
import numpy as np
import random
from megengine.data import transform as T
import matplotlib.pyplot as plt
import math
import model_cd as model
import pickle5
from dataloader import *


# In[2]:


net = model.Network()

# In[7]:
#def init_module(net):
#    for m in net:
#        if isinstance(m, M.Conv2d):
#            M.init.msra_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#            if m.bias is not None:
#                fan_in, _ = M.init.calculate_fan_in_and_fan_out(m.weight)
#                bound = 1 / math.sqrt(fan_in)
#                M.init.uniform_(m.bias, -bound, bound)
#        elif isinstance(m, M.BatchNorm2d):
#            M.init.ones_(m.weight)
#            M.init.zeros_(m.bias)
#        elif isinstance(m, M.Linear):
#            M.init.msra_uniform_(m.weight, a=math.sqrt(5))
#        if m.bias is not None:
#                fan_in, _ = M.init.calculate_fan_in_and_fan_out(m.weight)
#                bound = 1 / math.sqrt(fan_in)
#                M.init.uniform_(m.bias, -bound, bound)
#init_module(net.parameters())

pkl_file = open('model_adam.pkl', 'rb')
model_p = pickle5.load(pkl_file)
net.load_state_dict(model_p)

epochs = 1000
opt = mge.optimizer.Adam(net.parameters(), lr=1e-6)
gm = mge.autodiff.GradManager().attach(net.parameters())
train_epoch_loss, val_epoch_loss = [], []
min = 0.00206130
for epoch in range(epochs):
    for g in opt.param_groups:
        g['lr'] = (1e-6)*(1.05-F.sin(1.07*epoch/epochs))  # * (1-(epoch/300))
    train_losses, val_losses = [], []
    for batch, (features, labels) in enumerate(train_dataloader):
        opt.clear_grad()
        # features, labels = trans4(trans3(trans2(trans1((features, labels)))))
        features = mge.tensor(np.maximum(features.astype(np.float32) - 512, 0)/65535)
        labels = mge.tensor(labels.astype(np.float32)/65535)
        with gm:
            pred = net(features)
            train_loss = F.abs(pred - labels).mean()
            train_loss_np = float(train_loss.numpy())
            gm.backward(train_loss)
            opt.step()
        train_losses.append(train_loss_np)
        if batch % 20 == 0:
            print(f"epoch[{epoch+1}/{epochs}], batch[{batch+1}/114],loss: {np.mean(train_losses)}")
    for batch, (features, labels) in enumerate(val_dataloader):
        features = mge.tensor(np.maximum(features.astype(np.float32) - 512, 0)/65535)
        labels = mge.tensor(labels.astype(np.float32)/65535)
        pred = net(features)
        val_loss = F.abs(pred - labels).mean()
        val_loss = float(val_loss.numpy())
        val_losses.append(val_loss)
    print(f"epoch {epoch+1}, train loss: {np.mean(train_losses)},val loss: {np.mean(val_losses)}")
    train_epoch_loss.append(np.mean(train_losses))
    val_epoch_loss.append(np.mean(val_losses))
    if np.mean(val_losses) < min:
        min = np.mean(val_losses)
        fout = open('model_adam'+str(epoch)+str(np.mean(val_losses))+'.pkl', 'wb')
        mge.save(net.state_dict(), fout)
        fout.close()
    if (epoch + 1) % 500 == 0:
        fout = open('model_adam_'+str(epoch)+'.pkl', 'wb')
        mge.save(net.state_dict(), fout)
        fout.close()
fout = open('model_adam_'+str(epoch)+'.pkl', 'wb')
mge.save(net.state_dict(), fout)
fout.close()

# In[ ]:




