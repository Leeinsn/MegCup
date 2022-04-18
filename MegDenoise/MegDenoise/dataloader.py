import numpy as np
import megengine as mge
from megengine.data import DataLoader, RandomSampler
from megengine.data import SequentialSampler
import random

patchsz = 256
batch_size = 128

class Transpose(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, inputs):
        sample, sample_gt = inputs
        if random.random() < self.prob:
            return sample.transpose(0,1,3,2), sample_gt.transpose(0,1,3,2)
        return sample, sample_gt

class HorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, inputs):
        sample, sample_gt = inputs
        if random.random() < self.prob:
            return np.flip(sample, 2), np.flip(sample_gt, 2)
        return sample, sample_gt

class VerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, inputs):
        sample, sample_gt = inputs
        if random.random() < self.prob:
            return np.flip(sample, 1), np.flip(sample_gt, 1)
        return sample, sample_gt

class BrightnessContrast(object):
    def __init__(self, norm_num, prob=0.5):
        self.prob = prob
        self.norm_num = norm_num
    
    def __call__(self, inputs):
        sample, sample_gt = inputs
        h, w = sample.shape[1], sample.shape[2]
        if random.random() < self.prob:
            alpha = random.random() + 0.5
            beta = (random.random() * 150 + 50) / self.norm_num
            # np.full()构造数组用指定值填充
            bbeta = np.full((1, h, w), beta)
            sample = alpha * sample + bbeta
            if sample_gt is not None:
                sample_gt = alpha * sample_gt + bbeta
        return sample, sample_gt


print('loading data')
content = open('/temp_disk2/home/burst_raw/competition_train_input.0.2.bin', 'rb').read()
samples_ref = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))
content = open('/temp_disk2/home/burst_raw/competition_train_gt.0.2.bin', 'rb').read()
samples_gt = np.frombuffer(content, dtype='uint16').reshape((-1, 256, 256))
train_ref, train_gt, val_ref, val_gt = [], [], [], []
for i in range(0, 8192):
    if i < 8192 and i % 10 == 0:
        val_ref.append(samples_ref[i])
        val_gt.append(samples_gt[i])
    else:
        train_ref.append(samples_ref[i])
        train_gt.append(samples_gt[i])

val_ref = np.expand_dims(np.array(val_ref), axis=1)
val_gt = np.expand_dims(np.array(val_gt), axis=1)
train_ref = np.expand_dims(np.array(train_ref), axis=1)
train_gt = np.expand_dims(np.array(train_gt), axis=1)

print(val_ref.shape, val_gt.shape, train_ref.shape, train_gt.shape)

class train_dataset(mge.data.dataset.Dataset):
    def __init__(self):
        self.features, self.labels = train_ref, train_gt
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


train_dataset = train_dataset()
train_sampler = RandomSampler(dataset=train_dataset, batch_size=batch_size, drop_last=False)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler)


class val_dataset(mge.data.dataset.Dataset):
    def __init__(self):
        self.features, self.labels = val_ref, val_gt
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
val_dataset = val_dataset()
val_sampler = SequentialSampler(dataset=val_dataset, batch_size=batch_size, drop_last=False)
val_dataloader = DataLoader(val_dataset, sampler=val_sampler)
