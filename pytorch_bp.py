import numpy as np
import pandas as pd
import matplotlib
from pathlib import Path
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn, optim, hub

from torchvision import models, transforms, datasets

import imageio
from PIL import Image
from urllib.request import urlopen

torch.set_printoptions(edgeitems=2)
torch.manual_seed(123)

home = Path(r'C:\Users\jacob\Downloads\dlwpt-code-master')
data_url = 'https://raw.githubusercontent.com/deep-learning-with-pytorch/dlwpt-code/master/data'

#%%

# LOAD PRETRAINED IMG CLASS NETWORK

# Load model
alexnet = models.AlexNet()
resnet = models.resnet101(pretrained=True)

# Data pre-processing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )])


# Load test data
pic = f'{data_url}/p1ch2/bobby.jpg'
img = Image.open(urlopen(pic))
img_t = preprocess(img)

batch_t = torch.unsqueeze(img_t, 0)

# Initialise model
resnet.eval()

# Calculate output
out = resnet(batch_t)

# Lookup label
labels = f'{data_url}/p1ch2/imagenet_classes.txt'
f = urlopen(labels)

labels = [line.decode('utf-8').strip() for line in f.readlines()]

_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

#labels[index[0]], percentage[index[0]].item()

# Sort labels by confidence
_, indices = torch.sort(out, descending=True)
#[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]

#%%

# GAN

class ResNetBlock(nn.Module): # <1>

    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x) # <2>
        return out


class ResNetGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9): # <3> 

        assert(n_blocks >= 0)
        super(ResNetGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResNetBlock(ngf * mult)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input): # <3>
        return self.model(input)

# Instantiate model    
netG = ResNetGenerator()

# Load weights
model_path = f'{data_url}/p1ch2/horse2zebra_0.4.0.pth'
model_data = hub.load_state_dict_from_url(model_path)
netG.load_state_dict(model_data)

# Eval mode
netG.eval()

# Preprocessing pipe
preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.ToTensor()])

# Load image
#img = Image.open(urlopen(f'{data_url}/p1ch2/horse.jpg'))

# Preprocess image
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

# Compute image output
batch_out = netG(batch_t)

# Convert to image file
out_t = (batch_out.data.squeeze() + 1.0) / 2.0
out_img = transforms.ToPILImage()(out_t)
#out_img

#%%

# Loading model from internet

resnet18_model = hub.load('pytorch/vision:master',
                          'resnet18',
                          pretrained=True)

#%%

# Image tensor manipulation

img_arr = imageio.imread(f'{data_url}/p1ch4/image-dog/bobby.jpg')
img = torch.from_numpy(img_arr)
out = img.permute(2,0,1)

batch_size = 3
batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)

data_dir = home / 'data/p1ch4/image-cats'

for i, file in enumerate(data_dir.glob('*.png')):
    img_arr = imageio.imread(file)
    img_t = torch.from_numpy(img_arr)
    img_t = img_t.permute(2,0,1)
    img_t = img_t[:3]
    batch[i] = img_t
    
batch = batch.float()
batch /= 255.0

n_channels = batch.shape[1]

for c in range(n_channels):
    mean = torch.mean(batch[:,c])
    std = torch.std(batch[:,c])
    batch[:,c] = (batch[:,c] - mean)/std
    
#%%


dir_path = home / 'data/p1ch4/volumetric-dicom/2-LUNG 3.0  B70f-04083'
vol_arr = imageio.volread(dir_path, 'DICOM')
#vol_arr.shape

vol = torch.from_numpy(vol_arr).float()
vol = torch.unsqueeze(vol, 0)


#%%

# Tabular data

wine_path = home / 'data/p1ch4/tabular-wine/winequality-white.csv'
wine_df = pd.read_csv(wine_path, sep=';')

col_list = wine_df.columns.to_list()

wineq = torch.from_numpy(wine_df.values).float()

data = wineq[:,:-1]
target = wineq[:,-1].long()

# One-hot encoding

target_unsqueezed = target.unsqueeze(1)

target_onehot = torch.zeros(target.shape[0], 10)
target_onehot.scatter_(1, target_unsqueezed, 1.0)

data_mean = torch.mean(data, dim=0)
data_var = torch.var(data, dim=0)

data_normalized = (data - data_mean) / torch.sqrt(data_var)

# Classifier - binning scores
bad_data = data[target <= 3]
mid_data = data[(3 < target) & (target < 7)]
good_data = data[7 <= target]

bad_mean = torch.mean(bad_data, dim=0)
mid_mean = torch.mean(mid_data, dim=0)
good_mean = torch.mean(good_data, dim=0)

print(f'{"":20}   Bad    Mid    Good')
for col, bad, mid, good in zip(col_list, bad_mean, mid_mean, good_mean):
    print(f'{col:20} {bad:6.2f} {mid:6.2f} {good:6.2f}')

total_sulfur_threshold = 141.83
total_sulfur_data = data[:,6]
predicted_indices = torch.lt(total_sulfur_data, total_sulfur_threshold)

actual_indices = target > 5
n_matches = torch.sum(actual_indices & predicted_indices).item()
n_predicted = torch.sum(predicted_indices).item()
n_actual = torch.sum(actual_indices).item()


#%%

# TIME SERIES

labels = f'{data_url}/p1ch4/bike-sharing-dataset/hour-fixed.csv'
bikes_file = urlopen(labels)

bikes_df = pd.read_csv(bikes_file)
bikes_df['dteday'] = pd.to_datetime(bikes_df['dteday']).dt.day

bikes = torch.from_numpy(bikes_df.values).float()

# Transform 2d time series ino 3d time series of series (24hrs)
daily_bikes = bikes.view(-1, 24, bikes.shape[1])
daily_bikes = daily_bikes.transpose(1,2)

# One hot encode ordinal target variable
n_samples = daily_bikes.shape[0]
sample_length = daily_bikes.shape[2] # Number of hours recordings per sample (24)
n_levels = daily_bikes[:,9].unique().shape[0]

daily_weather_onehot = torch.zeros(n_samples, n_levels, sample_length)
daily_weather_onehot.scatter_(dim=1,
                              index=daily_bikes[:,9,].unsqueeze(1).long() - 1,
                              value=1.0)

# Concatenate X and y
daily_bikes = torch.cat((daily_bikes, daily_weather_onehot), dim=1)

# Scaling variables

daily_bikes[:,9,:] -= 1
daily_bikes[:,9,:] /= 3

temp = daily_bikes[:,10,:]

def normalise_1(x):
    x_min = torch.min(x)
    x_max = torch.max(x)
    x -= x_min
    x /= (x_max - x_min)
    
    return x

def normalise_2(x):
    return (x - torch.mean(x)) / torch.std(x)
    
temp = normalise_1(temp)

#%%

# TEXT

text = urlopen(f'{data_url}/p1ch4/jane-austen/1342-0.txt')
text = text.read().decode('utf8')
lines = text.split('\n')

line = lines[200]

letter_t = torch.zeros(len(line), 128) # ASCII

for i, letter in enumerate(line.lower().strip()):
    letter_index = ord(letter) if ord(letter) < 128 else 0
    letter_t[i][letter_index] = 1
    
def clean_words(input_str):
    punctuation = '.,;:"!?”“_-'
    word_list = input_str.lower().replace('\n',' ').split()
    word_list = [word.strip(punctuation) for word in word_list]
    return word_list

words_in_line = clean_words(line)

word_list = sorted(set(clean_words(text)))
word2index_dict = {word: i for (i, word) in enumerate(word_list)}


#%%

# LINEAR MODEL

# Plotting functions 
    
def plot_results(x, model, normalize):
    # Plotting
    fig, ax = plt.subplots(nrows=4, ncols=1, dpi=600) #x.shape[1]
    
    for i, axes in enumerate(ax.flatten()):
        axes.scatter(x[:,i+5],y)
        
        x_sorted, indices = x[:,i+5].sort()
        
        axes.plot(x_sorted, model(x[indices,:]).detach(), c='orange')
    
def plot_losses(train_losses, loss_fn, val_losses=None):
    # Plotting
    fig = plt.figure(dpi=600)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    n_epochs = len(train_losses)
    
    plt.plot(range(n_epochs), train_losses, label='Training loss')
    
    if val_losses:
        plt.plot(range(n_epochs), val_losses, label='Validation loss')
        
    plt.legend()
    
def pairplot_variables(x,y):
    n_vars = x.shape[1]
    n_vars=6
    
    fig, axes = plt.subplots(nrows=n_vars, ncols=n_vars, dpi=800)
    
    y = y.squeeze().numpy()
    n_classes = len(set(y))
    
    if n_classes < 20:
        cmap = plt.get_cmap('jet', n_classes)
    else:
        cmap = 'jet'
    
    for i in range(n_vars):
        for j in range(n_vars):
            if j < i:
                im = axes[i,j].scatter(x[:,i], x[:,j], c=y, cmap=cmap, s=0.1)
            else:
                axes[i,j].set_axis_off()
    
    fig.colorbar(im, ax=axes.flat)
    
# Training functions
    
def train_val_split(x, y, val_prop=0.2):
    n_samples = x.shape[0]
    n_val = int(val_prop * n_samples)
    
    shuffled_indices = torch.randperm(n_samples)
    
    train_indices = shuffled_indices[:-n_val]
    val_indices = shuffled_indices[-n_val:]
    
    x_train = x[train_indices]
    y_train = y[train_indices]
    
    x_val = x[val_indices]
    y_val = y[val_indices]
    
    return x_train, y_train, x_val, y_val

def calc_forward(x, y, model, loss_fn, is_train):
    with torch.set_grad_enabled(is_train):
        y_pred = model(x)
        loss = loss_fn(y_pred,y)
    return loss

def training_loop(x, y, model, loss_fn=nn.MSELoss(), optimizer=optim.SGD, 
                  n_epochs=5000, learning_rate=1e-2, normalize=True, val_split=True,
                  batch_size=None):
    
    # Instantiating optimizer
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    
    # Train/Val split
    if val_split:
        x_train, y_train, x_val, y_val = train_val_split(x, y)
    else:
        x_train, y_train = x, y
    
    if normalize:
        # TODO
        pass
    
    # Storage for plotting efficiency
    val_losses = []
    train_losses = []
    
    # Loading batches
    if not batch_size:
        batch_size = x.shape[0]
    
    train_loader = torch.utils.DataLoader(x, batch_size, shuffle=True)
        
    
    for epoch in range(n_epochs):
        
        # Training
        for x_batch, y_batch in train_loader:
            train_loss = calc_forward(x_batch, y_batch, model, loss_fn, is_train=True) #Forward pass
            train_losses.append(float(train_loss))
            
            # Backward pass / param update
            optimizer.zero_grad()
            train_loss.backward() #Backward pass (calculating the derivative)
            optimizer.step()
            
        # Validation
        if val_split:
            val_loss = calc_forward(x_val, y_val, model, loss_fn, is_train=False)
            val_losses.append(float(val_loss))
        else:
            val_losses = None
        
        if epoch % 1 == 0: #000 == 0:
            print(f'(Epoch {epoch}) Training loss: {float(train_loss):.4f}'.rjust(35))
            if val_split:
                print(f'Validation loss: {float(val_loss):.4f}'.rjust(35))
                
        elif epoch % 1001 == 0:
            print('...\n')

        if not torch.isfinite(train_loss).all():
            break
    
    plot_results(x, model, normalize)
    plot_losses(train_losses, loss_fn, val_losses)
    
    return model.parameters()

def seq_model(dim_input, dim_layer_1, dim_output, final_layer=None):
    
    if final_layer:
        model = nn.Sequential(nn.Linear(dim_input, dim_layer_1),
                          nn.Tanh(),
                          nn.Linear(dim_layer_1, dim_output),
                          final_layer)
    else:
        model = nn.Sequential(nn.Linear(dim_input, dim_layer_1),
                              nn.Tanh(),
                              nn.Linear(dim_layer_1, dim_output))
        
    return model


#%%
# Temp data
    
temp_celsius = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
temp_unknown = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]

y = torch.tensor(temp_celsius).unsqueeze(1)
x = torch.tensor(temp_unknown).unsqueeze(1)

# Wine data

x = wineq[:,:-1]
y = wineq[:,-1]

# Frame problem
predict_class = True

if predict_class:
    y = y.long() #.unsqueeze(1)
    loss_fn = nn.CrossEntropyLoss()
    dim_output = torch.max(y).item() + 1

else:
    # Onehot (class as opposed to regr)
    #y_class = torch.zeros(y.shape[0], 10)
    #y_class.scatter_(1, y, 1.0)
    y = y.float().unsqueeze(1)
    loss_fn = nn.MSELoss()
    dim_output = y.shape[1]

# Model

dim_input = x.shape[1]
dim_layer_1 = 100

model = seq_model(dim_input, dim_layer_1, dim_output)

# Train

train = True

if train:
    params = training_loop(
                    x = x,
                    y = y,
                    optimizer = optim.Adam,
                    learning_rate = 1e-3,
                    loss_fn = loss_fn,
                    n_epochs = 1000,
                    normalize = False,
                    model = model
                    )

# TODO
# One hot encode output            
if not predict_class:
    pairplot_variables(x, y)

#%%

# IMAGES

data_path = home / 'data/p1ch7'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)

img, label = cifar10[99]
#plt.imshow(img)

to_tensor = transforms.ToTensor()
img_t = to_tensor(img)

tensor_cifar10 = datasets.CIFAR10(data_path, train=True, download=False,
                                  transform=transforms.ToTensor())

img_t, label = tensor_cifar10[99]
imgs = torch.stack([img_t for img_t, _ in tensor_cifar10], dim=3)

train_mean = imgs.view(3,-1).mean(dim=1)
train_std = imgs.view(3,-1).std(dim=1)

preprocessing = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(train_mean, train_std)])

transformed_cifar10 = datasets.CIFAR10(data_path, train=True, download=False, 
                                       transform=preprocessing)

transformed_cifar10_val = datasets.CIFAR10(data_path, train=False, download=False, 
                                           transform=preprocessing)


to_pil = transforms.ToPILImage()
img_t, _ = transformed_cifar10[99]

# ToPILImage has weird behaviour if values outside of RGB range
# Directly permuting cols just hardcuts at black/white if above 1/below 0
plt.imshow(to_pil(img_t))
#plt.imshow(img_t.permute(1,2,0))

#%%

# Image classifier net

label_map = {0:0, 2:1}
class_names = ['aeroplane', 'bird']
cifar2 = [(img, label_map[label]) for img, label in transformed_cifar10 if label in [0,2]]
cifar2_val = [(img, label_map[label]) for img, label in transformed_cifar10_val if label in [0,2]]

#%%

cifar2_small = cifar2

x = torch.cat([pair[0].view(-1).unsqueeze(0) for pair in cifar2_small], dim=0)
y = torch.tensor([pair[1] for pair in cifar2_small])
loss_fn = nn.NLLLoss()

dim_input = x.shape[1]
dim_output = len(class_names) #y.unique.shape[0]
dim_layer_1 = 512

model = seq_model(dim_input, 
                  dim_layer_1, 
                  dim_output, 
                  final_layer=nn.LogSoftmax(dim=1))

train = True

if train:
    params = training_loop(
                    x = x,
                    y = y,
                    optimizer = optim.SGD,
                    learning_rate = 1e-2,
                    loss_fn = loss_fn,
                    n_epochs = 100,
                    normalize = False,
                    model = model,
                    val_split=False,
                    batch_size=1
                    )
    
