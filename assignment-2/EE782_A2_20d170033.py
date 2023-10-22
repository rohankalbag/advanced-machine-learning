# %% [markdown]
# # EE 782 Assignment 2
# ## Name: Rohan Rajesh Kalbag
# ## Roll No: 20d170033

# %% [markdown]
# ## Importing Libraries

# %%
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import PIL
import torchvision
import matplotlib.pyplot as plt

# %% [markdown]
# ## Q1 : Getting the labelled face dataset from specified source

# %%
# running this locally on my CUDA enabled machine which has Nvidia RTX 2070
!nvidia-smi # check GPU

# %%
%%bash
# download the dataset if not already downloaded
if [ ! -d "lfw" ]; then
    wget http://vis-www.cs.umass.edu/lfw/lfw.tgz # download the dataset
    tar -xzf lfw.tgz # extract the dataset
    rm lfw.tgz # remove the tar file
fi

# %% [markdown]
# ## Q2 : Getting the names of people with more than one image and also the number

# %%
import os

# list to hold people with more than one image
people_with_more_than_one_image = []

# Define the root directory
root_directory = "./lfw"

# Create a dictionary to store subfolder file counts
subfolder_counts = {}

# Walk through the directory and count files in subfolders
for dirpath, _, filenames in os.walk(root_directory):
    subfolder_name = os.path.basename(dirpath) # get the subfolder name
    if(subfolder_name in subfolder_counts.keys()): # check if key exists
      subfolder_counts[subfolder_name] += len(filenames) # add count to existing key
    else:
      subfolder_counts[subfolder_name] = len(filenames) # add new key and count

# Get subfolders with more than one file
people_with_more_than_one_image = [folder for folder, count in subfolder_counts.items() if count > 1]

# %%
print(len(people_with_more_than_one_image)) # print number of people with more than one image
print(people_with_more_than_one_image[:30]) # print first 30 people with more than one image for reference

# %% [markdown]
# Thus we see that there are 1680 people in the dataset who have more than one image, and the first 30 of their names are printed below

# %% [markdown]
# # Part A

# %% [markdown]
# ## Q3 : Splitting into Test, Train and Validation by Person
# 
# ### We initially split the dataset into 70% train and 15% test and 15% validation

# %%
import numpy as np
import random

# shuffle the names
random.shuffle(people_with_more_than_one_image)

# Defining the proportions for train, validation, and test sets
# for now assuming
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15

# Calculate the sizes for each set
total_samples = len(people_with_more_than_one_image)
train_size = int(train_ratio * total_samples)
validation_size = int(validation_ratio * total_samples)
test_size = total_samples - train_size - validation_size

# Use NumPy to split the data
train_data_names = people_with_more_than_one_image[:train_size]
validation_data_names = people_with_more_than_one_image[train_size:train_size + validation_size]
test_data_names = people_with_more_than_one_image[train_size + validation_size:]

# Now you have your data split into train, validation, and test sets
print("Number of People in Train Data:", len(train_data_names))
print("Number of People in Validation Data:", len(validation_data_names))
print("Number of People in Test Data:", len(test_data_names))

# %%
class SiameseNetworkDataset(Dataset):

    def __init__(self, personNames, dataset_size, transform=None, should_invert = False):
        self.personNames = personNames # list of people names
        self.transform = transform # transform to apply to images
        self.should_invert = should_invert # whether to invert images
        self.dataset_size = dataset_size

    def __getitem__(self, index):
        person1 = random.choice(self.personNames) # get a random person name
        person2 = random.choice(self.personNames) # get another random person name

        while person1 == person2: # make sure the two names are not the same
          person2 = random.choice(self.personNames)

        should_get_same_class = random.randint(0, 1) # randomly decide whether to get images of same person or not

        if should_get_same_class: # if same person
           img0_name = random.choice(os.listdir(root_directory + f'/{person1}'))
           img1_name = random.choice(os.listdir(root_directory + f'/{person1}'))
           person2 = person1 # set person2 to person1
        else: # if different person
            img0_name = random.choice(os.listdir(root_directory + f'/{person1}'))
            img1_name = random.choice(os.listdir(root_directory + f'/{person2}'))

        img0 = PIL.Image.open(root_directory + f'/{person1}' + f'/{img0_name}')
        img1 = PIL.Image.open(root_directory + f'/{person2}' + f'/{img1_name}')
        img0 = img0.convert("RGB") # convert to RGB
        img1 = img1.convert("RGB")

        if self.should_invert: # invert images if specified
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None: # apply transform if specified
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        if person1 == person2:
          label = 1.0 # if same person
        else:
          label = -1.0 # if different person

        return img0, img1, torch.from_numpy(np.array(label, dtype=np.float32)) # return images and label

    def __len__(self):
        return self.dataset_size

# %% [markdown]
# ### Let us visualize the working of the Custom Dataset created to train our Siaamese Network

# %%
visualize_dataset = SiameseNetworkDataset(personNames = train_data_names, dataset_size = 1000, transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))

# %%
visualize_dataloader = DataLoader(visualize_dataset, shuffle=True, batch_size=8)

# %%
number_of_batches_visualized = 5
for batch in range(number_of_batches_visualized):
  print(f"Batch : {batch}") # print batch number
  dataiter = iter(visualize_dataloader) # get a batch
  batch = next(dataiter) # get a batch
  print(batch[2].numpy().T) # print labels
  concatenated = torch.cat((batch[0], batch[1]), 0)
  img = torchvision.utils.make_grid(concatenated) # concatenate images
  npimg = img.numpy()
  # visualize the batch
  plt.figure(figsize = (16, 8))
  plt.axis("off")
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

# %% [markdown]
# Thus we see that our dataset is creating similar and dissimilar pairs of images and also the labels for them correctly. It creates a label of 1 for similar images and -1 for dissimilar images.

# %% [markdown]
# ## Q4: Selecting a pre trained model trained on ImageNet keeping in mind the tradeoff of computational resources as well as accuracy
# 
# - We will be using Pytorch's ResNet-18 which is pre trained on ImageNet for this application

# %%
resnet50_model = models.resnet50(weights='ResNet50_Weights.DEFAULT') # load the ResNet50 model which has been trained on ImageNet for Transfer Learning

# %% [markdown]
# ## Q5: Appropriately crop and resize the images based on your computational resources
# - We reshape our images to 224x224x3 using the transform operation which we defined earlier for the dataset class
# - We also choose the batch sizes, dataset sizes for the train, test and validation datasets

# %%
train_dataset = SiameseNetworkDataset(personNames=train_data_names, dataset_size=1000, transform=transforms.Compose([
                                      transforms.Resize((224, 224)), transforms.ToTensor()]))
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=48)

valid_dataset = SiameseNetworkDataset(personNames=validation_data_names, dataset_size=100, transform=transforms.Compose([
                                      transforms.Resize((224, 224)), transforms.ToTensor()]))
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=8)

test_dataset = SiameseNetworkDataset(personNames=test_data_names, dataset_size=100, transform=transforms.Compose([
                                     transforms.Resize((224, 224)), transforms.ToTensor()]))
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=8)

# %% [markdown]
# ## Q6: Defining the Siamese Network
# - We use the ResNet model as the base model for our Siamese Network
# - We drop the softmax layer of the ResNet model and add a fully connected layer with three hidden layers, we also use RELU activation for the hidden layers, the last layer output will not have any activation function and will serve as embedding for the images
# - We use transfer learning to train the model, we freeze the weights of the ResNet model and only train the weights of the fully connected layers

# %%
class SiameseNetwork(nn.Module):

    def __init__(self, base_model):
        super(SiameseNetwork, self).__init__()

        self.base_model = nn.Sequential(*list(base_model.children())[:-1]) # get the base model ResNet50
        # remove the last layer of ResNet50 which is the Softmax layer

        for wt in self.base_model.parameters():
            wt.requires_grad_(False) # freeze the weights of the base model ResNet50

        # fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128), # 128 dimensional embedding
        )

    def forward_one(self, x):
        # forward pass of one image
        x = self.base_model(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1) # forward pass of first image
        output2 = self.forward_one(input2) # forward pass of second image
        return output1, output2 # return the two embeddings

# %% [markdown]
# ## Q6:  Metric Learning Scheme (cosine similarity or Euclidean distance, paired with cross-entropy or hinge loss with a margin)
# - We use PyTorch's [`CosineEmbeddingLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html) as our loss function, which internally handles both cosine similarity paired with a hinge with user defined margin as the loss function.

# %%
if torch.cuda.is_available():
    model = SiameseNetwork(resnet50_model).cuda() # use GPU if available
else:
    model = SiameseNetwork(resnet50_model)
criterion = torch.nn.CosineEmbeddingLoss() # use Cosine Embedding Loss
optimizer = optim.Adagrad(model.parameters(), lr=0.01) # use Adagrad optimizer

# %% [markdown]
# ## Training the Model
# - we track the running training loss, and in order to perform the validation we choose a threshold 0.8 for the cosine similarity, if the cosine similarity is greater than 0.8 we consider the images to be similar, else we consider them to be dissimilar, and thus we track the running validation accuracy

# %%
def train_model(model, number_of_epochs, visualize=False):
    threshold = 0.8 # threshold for cosine similarity above which two images are considered to be of the same person
    curr_epoch = 0
    counter = []
    train_loss_history = []
    valid_accuracy_history = []
    for epoch in range(number_of_epochs):
        avg_train_loss = 0.0 # average training loss
        for i, data in enumerate(train_dataloader):
            img0, img1, label = data
            optimizer.zero_grad() # zero the gradients

            if torch.cuda.is_available():
                output1, output2 = model(img0.cuda(), img1.cuda())
            else:
                output1, output2 = model(img0, img1)

            if torch.cuda.is_available():
                loss = criterion(output1, output2, label.cuda())
            else:
                loss = criterion(output1, output2, label)

            loss.backward() # backpropagate the loss
            optimizer.step() # update the weights
            avg_train_loss += loss.item() # add the loss to the average training loss
        avg_train_loss /= len(train_dataloader)

        correct = 0
        # validation
        for j, vdata in enumerate(valid_dataloader):
            vimg0, vimg1, vlabel = vdata

            if torch.cuda.is_available():
                voutput1, voutput2 = model(vimg0.cuda(), vimg1.cuda())
            else:
                voutput1, voutput2 = model(vimg0, vimg1)

            cosine_similarity = torch.nn.functional.cosine_similarity(
                voutput1, voutput2)
            pred_label = cosine_similarity.cpu().detach().numpy()
            pred_label[pred_label > threshold] = 1.0 # if cosine similarity is above threshold, images are of same person
            pred_label[pred_label <= threshold] = -1.0 # if cosine similarity is below threshold, images are of different people
            correct += np.sum(pred_label == vlabel.cpu().detach().numpy()) # add the number of correct predictions

        accuracy = correct*100/(8*len(valid_dataloader)) # calculate the accuracy denominator is 8 because batch size is 8

        print("Epoch number {} : Training loss {} : Validation Accuracy {}%".format(
            epoch, avg_train_loss, accuracy))
        curr_epoch += 1
        counter.append(curr_epoch)
        train_loss_history.append(avg_train_loss)
        valid_accuracy_history.append(accuracy)
    
    if (visualize): # visualize the training loss and validation accuracy
        plt.plot(counter, train_loss_history)
        plt.xlabel("Epochs")
        plt.ylabel("Training Loss")
        plt.show()
        plt.plot(counter, valid_accuracy_history)
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy")
        plt.show()

# %% [markdown]
# ## Without Regularization and Image Augmentation

# %%
train_model(model, 75, visualize=True)

# %% [markdown]
# - We see that the training loss is decreasing and the moving average of our validation accuracy is increasing, thus we can say that our model is learning

# %%
torch.save(model, 'epochs75.pt')

# %%
model = torch.load('epochs75.pt')

# %% [markdown]
# ### Cosine Similarity Visualization

# %%
dataiter = iter(test_dataloader)
for i in range(3):
    img0, img1, label = next(dataiter)
    concatenated = torch.cat((img0, img1),0)

    output1, output2 = model(img0.cuda(), img1.cuda()) # forward pass
    cosine_similarity = torch.nn.functional.cosine_similarity(output1, output2) # calculate cosine similarity
    print(cosine_similarity.cpu().detach().numpy().T) # print cosine similarity
    img = torchvision.utils.make_grid(concatenated)
    npimg = img.numpy()
    # visualize the batch
    plt.figure(figsize = (16, 8))
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# %% [markdown]
# As we can see that the cosine similarities of the embeddings of the similar images are close to 1 and the cosine similarities of the embeddings of the dissimilar images further away from 1 closer to 0 and -1

# %% [markdown]
# ### Test Accuracy

# %%
# same code as in training used for validation accuracy, but now used for test accuracy
threshold = 0.8
correct = 0
for i, data in enumerate(test_dataloader):
  img0, img1, label = data

  if torch.cuda.is_available():
    output1, output2 = model(img0.cuda(), img1.cuda())
  else:
    output1, output2 = model(img0, img1)

  cosine_similarity = torch.nn.functional.cosine_similarity(output1, output2)
  pred_label = cosine_similarity.cpu().detach().numpy()
  pred_label[pred_label > threshold] = 1.0
  pred_label[pred_label <= threshold] = -1.0

  correct += np.sum(pred_label == label.cpu().detach().numpy())

print("Accuracy:", correct*100/(8*len(test_dataloader)))

# %% [markdown]
# ## 6 (a) With Image Augmentation
# 
# - One out of random horizontal flip and random rotation of upto 10 degrees, also random color jitter (the brightness, contrast and saturation changed), gaussian blur, horizontal flip is applied at random with a probability of 0.2
# 
# - We keep the same model architecture and hyperparameters as before, also optimiser and loss function are kept the same so that we can compare the results

# %%
augmentation_transform = transforms.Compose([
    transforms.Resize((224, 224)), # resize to 224x224
    transforms.RandomApply([ # apply random transformations
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10), # rotate by 10 degrees
        transforms.ColorJitter(brightness=0.05, contrast=0.05, # change brightness, contrast, saturation, hue
                           saturation=0.05, hue=0.05),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.2)), # apply gaussian blur
    ], p=0.2), transforms.ToTensor()])

# %%
train_dataset = SiameseNetworkDataset(personNames=train_data_names, dataset_size=1000, transform=augmentation_transform)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=48)

valid_dataset = SiameseNetworkDataset(personNames=validation_data_names, dataset_size=100, transform=transforms.Compose([
                                      transforms.Resize((224, 224)), transforms.ToTensor()]))
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=8)

test_dataset = SiameseNetworkDataset(personNames=test_data_names, dataset_size=100, transform=transforms.Compose([
                                     transforms.Resize((224, 224)), transforms.ToTensor()]))
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=8)

# %%
if torch.cuda.is_available():
    model = SiameseNetwork(resnet50_model).cuda()
else:
    model = SiameseNetwork(resnet50_model)
criterion = torch.nn.CosineEmbeddingLoss()

optimizer = optim.Adagrad(model.parameters(), lr=0.01)

# %%
train_model(model, 75, visualize=True)

# %% [markdown]
# - We see that the training loss is decreasing and the moving average of our validation accuracy is increasing, thus we can say that our model is learning

# %%
torch.save(model, 'epochs75_aug.pt')

# %%
model = torch.load('epochs75_aug.pt')

# %% [markdown]
# ### Cosine Similarity Visualization

# %%
dataiter = iter(test_dataloader)
for i in range(3):
    img0, img1, label = next(dataiter)
    concatenated = torch.cat((img0, img1),0)

    output1, output2 = model(img0.cuda(), img1.cuda())
    cosine_similarity = torch.nn.functional.cosine_similarity(output1, output2)
    print(cosine_similarity.cpu().detach().numpy().T)
    img = torchvision.utils.make_grid(concatenated)
    npimg = img.numpy()
    plt.figure(figsize = (16, 8))
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# %% [markdown]
# As we can see that the cosine similarities of the embeddings of the similar images are close to 1 and the cosine similarities of the embeddings of the dissimilar images further away from 1 closer to 0 and -1

# %% [markdown]
# ### Test Accuracy

# %%
threshold = 0.8
correct = 0
for i, data in enumerate(test_dataloader):
  img0, img1, label = data

  if torch.cuda.is_available():
    output1, output2 = model(img0.cuda(), img1.cuda())
  else:
    output1, output2 = model(img0, img1)

  cosine_similarity = torch.nn.functional.cosine_similarity(output1, output2)
  pred_label = cosine_similarity.cpu().detach().numpy()
  pred_label[pred_label > threshold] = 1.0
  pred_label[pred_label <= threshold] = -1.0

  correct += np.sum(pred_label == label.cpu().detach().numpy())

print("Accuracy:", correct*100/(8*len(test_dataloader)))

# %% [markdown]
# 

# %% [markdown]
# ## 6 (b) With Regularization
# 

# %%
train_dataset = SiameseNetworkDataset(personNames=train_data_names, dataset_size=1000, transform=transforms.Compose([
                                      transforms.Resize((224, 224)), transforms.ToTensor()]))
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=48)

valid_dataset = SiameseNetworkDataset(personNames=validation_data_names, dataset_size=100, transform=transforms.Compose([
                                      transforms.Resize((224, 224)), transforms.ToTensor()]))
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=8)

test_dataset = SiameseNetworkDataset(personNames=test_data_names, dataset_size=100, transform=transforms.Compose([
                                     transforms.Resize((224, 224)), transforms.ToTensor()]))
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=8)

# %% [markdown]
# ### We use same architecture as before, but we add dropout layers with a probability of 0.15 after each hidden layer and also add L2 regularization with a weight decay of 0.0001

# %%
class SiameseNetwork(nn.Module):

    def __init__(self, base_model):
        super(SiameseNetwork, self).__init__()

        self.base_model = nn.Sequential(*list(base_model.children())[:-1])

        for wt in self.base_model.parameters():
            wt.requires_grad_(False)

        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.15), # add dropout
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.15), # add dropout
            nn.Linear(256, 128),
        )

    def forward_one(self, x):
        x = self.base_model(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

# %%
if torch.cuda.is_available():
    model = SiameseNetwork(resnet50_model).cuda()
else:
    model = SiameseNetwork(resnet50_model)
criterion = torch.nn.CosineEmbeddingLoss()

optimizer = optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0.0001) # add weight decay

# %%
train_model(model, 75, visualize=True)

# %% [markdown]
# - We see that the training loss is decreasing and the moving average of our validation accuracy is increasing, thus we can say that our model is learning

# %%
torch.save(model, 'epochs75_reg.pt')

# %%
model = torch.load('epochs75_reg.pt')

# %% [markdown]
# ### Cosine Similarity Visualization

# %%
dataiter = iter(test_dataloader)
for i in range(3):
    img0, img1, label = next(dataiter)
    concatenated = torch.cat((img0, img1),0)

    output1, output2 = model(img0.cuda(), img1.cuda())
    cosine_similarity = torch.nn.functional.cosine_similarity(output1, output2)
    print(cosine_similarity.cpu().detach().numpy().T)
    img = torchvision.utils.make_grid(concatenated)
    npimg = img.numpy()
    plt.figure(figsize = (16, 8))
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# %% [markdown]
# As we can see that the cosine similarities of the embeddings of the similar images are close to 1 and the cosine similarities of the embeddings of the dissimilar images further away from 1 closer to 0 and -1

# %% [markdown]
# ### Test Accuracy

# %%
threshold = 0.8
correct = 0
for i, data in enumerate(test_dataloader):
  img0, img1, label = data

  if torch.cuda.is_available():
    output1, output2 = model(img0.cuda(), img1.cuda())
  else:
    output1, output2 = model(img0, img1)

  cosine_similarity = torch.nn.functional.cosine_similarity(output1, output2)
  pred_label = cosine_similarity.cpu().detach().numpy()
  pred_label[pred_label > threshold] = 1.0
  pred_label[pred_label <= threshold] = -1.0

  correct += np.sum(pred_label == label.cpu().detach().numpy())

print("Accuracy:", correct*100/(8*len(test_dataloader)))

# %% [markdown]
# - Seeing the improvement in the test accuracy we can say that the model is learning better with regularization as well as image augmentation, so for all tasks from now on we will use regularization and image augmentation.

# %% [markdown]
# ## Q7 : Using Learning Rate Schedulers

# %%
augmentation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.05, contrast=0.05,
                           saturation=0.05, hue=0.05),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.2)),
    ], p=0.2), transforms.ToTensor()])

# %%
train_dataset = SiameseNetworkDataset(personNames=train_data_names, dataset_size=1000, transform=augmentation_transform)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=48)

valid_dataset = SiameseNetworkDataset(personNames=validation_data_names, dataset_size=100, transform=transforms.Compose([
                                      transforms.Resize((224, 224)), transforms.ToTensor()]))
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=8)

test_dataset = SiameseNetworkDataset(personNames=test_data_names, dataset_size=100, transform=transforms.Compose([
                                     transforms.Resize((224, 224)), transforms.ToTensor()]))
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=8)

# %%
class SiameseNetwork(nn.Module):

    def __init__(self, base_model):
        super(SiameseNetwork, self).__init__()

        self.base_model = nn.Sequential(*list(base_model.children())[:-1])

        for wt in self.base_model.parameters():
            wt.requires_grad_(False)

        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
        )

    def forward_one(self, x):
        x = self.base_model(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

# %%
if torch.cuda.is_available():
    model = SiameseNetwork(resnet50_model).cuda()
else:
    model = SiameseNetwork(resnet50_model)
criterion = torch.nn.CosineEmbeddingLoss()

optimizer = optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0.0001)

# %% [markdown]
# 
# ## StepLR
# - We use the StepLR scheduler with step size 15 and gamma 0.5, which will half the learning rate every 10 epochs

# %%
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5) # add learning rate scheduler

# %% [markdown]
# ### Modification to training function to include scheduler

# %%
def train_model_with_scheduler(model, number_of_epochs, scheduler, visualize=False):
    # modified training function to use scheduler
    threshold = 0.8
    curr_epoch = 0
    counter = []
    train_loss_history = []
    valid_accuracy_history = []
    for epoch in range(number_of_epochs):
        avg_train_loss = 0.0
        for i, data in enumerate(train_dataloader):
            img0, img1, label = data
            optimizer.zero_grad()

            if torch.cuda.is_available():
                output1, output2 = model(img0.cuda(), img1.cuda())
            else:
                output1, output2 = model(img0, img1)

            if torch.cuda.is_available():
                loss = criterion(output1, output2, label.cuda())
            else:
                loss = criterion(output1, output2, label)

            loss.backward()
            optimizer.step()
            avg_train_loss += loss.item()
        avg_train_loss /= len(train_dataloader)

        correct = 0

        for j, vdata in enumerate(valid_dataloader):
            vimg0, vimg1, vlabel = vdata

            if torch.cuda.is_available():
                voutput1, voutput2 = model(vimg0.cuda(), vimg1.cuda())
            else:
                voutput1, voutput2 = model(vimg0, vimg1)

            cosine_similarity = torch.nn.functional.cosine_similarity(
                voutput1, voutput2)
            pred_label = cosine_similarity.cpu().detach().numpy()
            pred_label[pred_label > threshold] = 1.0
            pred_label[pred_label <= threshold] = -1.0
            correct += np.sum(pred_label == vlabel.cpu().detach().numpy())

        accuracy = correct*100/(8*len(valid_dataloader))
        
        print("Epoch number {} : Training loss {} : Validation Accuracy {}% : Current LR {}".format(
            epoch, avg_train_loss, accuracy, scheduler.get_last_lr()))
        
        scheduler.step()
        curr_epoch += 1
        counter.append(curr_epoch)
        train_loss_history.append(avg_train_loss)
        valid_accuracy_history.append(accuracy)
    
    if (visualize):
        plt.plot(counter, train_loss_history)
        plt.xlabel("Epochs")
        plt.ylabel("Training Loss")
        plt.show()
        plt.plot(counter, valid_accuracy_history)
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy")
        plt.show()

# %%
train_model_with_scheduler(model, 75, scheduler, visualize=True)

# %%
torch.save(model, 'epochs75_aug_reg_steplr.pt')

# %%
model = torch.load('epochs75_aug_reg_steplr.pt')

# %% [markdown]
# ### Cosine Similarity Visualization

# %%
dataiter = iter(test_dataloader)
for i in range(3):
    img0, img1, label = next(dataiter)
    concatenated = torch.cat((img0, img1),0)

    output1, output2 = model(img0.cuda(), img1.cuda())
    cosine_similarity = torch.nn.functional.cosine_similarity(output1, output2)
    print(cosine_similarity.cpu().detach().numpy().T)
    img = torchvision.utils.make_grid(concatenated)
    npimg = img.numpy()
    plt.figure(figsize = (16, 8))
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# %% [markdown]
# As we can see that the cosine similarities of the embeddings of the similar images are close to 1 and the cosine similarities of the embeddings of the dissimilar images further away from 1 closer to 0 and -1

# %% [markdown]
# ### Test Accuracy

# %%
threshold = 0.8
correct = 0
for i, data in enumerate(test_dataloader):
  img0, img1, label = data

  if torch.cuda.is_available():
    output1, output2 = model(img0.cuda(), img1.cuda())
  else:
    output1, output2 = model(img0, img1)

  cosine_similarity = torch.nn.functional.cosine_similarity(output1, output2)
  pred_label = cosine_similarity.cpu().detach().numpy()
  pred_label[pred_label > threshold] = 1.0
  pred_label[pred_label <= threshold] = -1.0

  correct += np.sum(pred_label == label.cpu().detach().numpy())

print("Accuracy:", correct*100/(8*len(test_dataloader)))

# %% [markdown]
# - This doesn't perform as well as the constant learning rate one, however is able to achieve a test accuracy better as compared to the vanilla siamese network without regularization and image augmentation

# %% [markdown]
# 
# ## PolynomialLR
# - We use the Polynomial LR scheduler with total iters 150, power of 1, which will linearly decrease the learning rate from 0.01 to 0.005 over the course of 75 epochs

# %%
if torch.cuda.is_available():
    model = SiameseNetwork(resnet50_model).cuda()
else:
    model = SiameseNetwork(resnet50_model)
criterion = torch.nn.CosineEmbeddingLoss()

optimizer = optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0.0001)

# %%
scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=150, power=1) # use polynomial learning rate scheduler

# %%
train_model_with_scheduler(model, 75, scheduler, visualize=True)

# %%
torch.save(model, 'epochs75_aug_reg_polylr.pt')

# %%
model = torch.load('epochs75_aug_reg_polylr.pt')

# %% [markdown]
# ### Cosine Similarity Visualization

# %%
dataiter = iter(test_dataloader)
for i in range(3):
    img0, img1, label = next(dataiter)
    concatenated = torch.cat((img0, img1),0)

    output1, output2 = model(img0.cuda(), img1.cuda())
    cosine_similarity = torch.nn.functional.cosine_similarity(output1, output2)
    print(cosine_similarity.cpu().detach().numpy().T)
    img = torchvision.utils.make_grid(concatenated)
    npimg = img.numpy()
    plt.figure(figsize = (16, 8))
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# %% [markdown]
# As we can see that the cosine similarities of the embeddings of the similar images are close to 1 and the cosine similarities of the embeddings of the dissimilar images further away from 1 closer to 0 and -1

# %% [markdown]
# ### Test Accuracy

# %%
threshold = 0.8
correct = 0
for i, data in enumerate(test_dataloader):
  img0, img1, label = data

  if torch.cuda.is_available():
    output1, output2 = model(img0.cuda(), img1.cuda())
  else:
    output1, output2 = model(img0, img1)

  cosine_similarity = torch.nn.functional.cosine_similarity(output1, output2)
  pred_label = cosine_similarity.cpu().detach().numpy()
  pred_label[pred_label > threshold] = 1.0
  pred_label[pred_label <= threshold] = -1.0

  correct += np.sum(pred_label == label.cpu().detach().numpy())

print("Accuracy:", correct*100/(8*len(test_dataloader)))

# %% [markdown]
# - This doesn't perform as well as constant learning rate with image augmentation and regularization, however is able to achieve a test accuracy better as compared to the vanilla siamese network as well as the stepLR one with regularization and image augmentation

# %% [markdown]
# #### Thus we see that both the learning rate schedulers are able to achieve good test accuracy results as compared to the vanilla siamese network, the polynomial LR scheduler performs better than the stepLR scheduler. However the constant learning rate with image augmentation and regularization performs the best.
# 
# #### This may be because the StepLR leads to a sharper drop in learning rate per epoch and thus the model is not able to learn as well as the constant learning rate one, the polynomial LR scheduler is able to achieve better results than the stepLR scheduler because it decreases the learning rate linearly and thus the model is able to learn better. 

# %% [markdown]
# ## Q8: Using Different Optimizers
# 
# - Since we have already used the Adagrad optimizer, we will now use Adam and compare the results. We will pick the model which has the best test accuracy and use it for the next tasks. So we will compare the results of the constant learning rate with image augmentation and regularization with Adam and Adagrad optimizers.

# %%
augmentation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.05, contrast=0.05,
                           saturation=0.05, hue=0.05),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.2)),
    ], p=0.2), transforms.ToTensor()])

# %%
train_dataset = SiameseNetworkDataset(personNames=train_data_names, dataset_size=1000, transform=augmentation_transform)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=48)

valid_dataset = SiameseNetworkDataset(personNames=validation_data_names, dataset_size=100, transform=transforms.Compose([
                                      transforms.Resize((224, 224)), transforms.ToTensor()]))
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=8)

test_dataset = SiameseNetworkDataset(personNames=test_data_names, dataset_size=100, transform=transforms.Compose([
                                     transforms.Resize((224, 224)), transforms.ToTensor()]))
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=8)

# %%
class SiameseNetwork(nn.Module):

    def __init__(self, base_model):
        super(SiameseNetwork, self).__init__()

        self.base_model = nn.Sequential(*list(base_model.children())[:-1])

        for wt in self.base_model.parameters():
            wt.requires_grad_(False)

        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
        )

    def forward_one(self, x):
        x = self.base_model(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

# %%
if torch.cuda.is_available():
    model = SiameseNetwork(resnet50_model).cuda()
else:
    model = SiameseNetwork(resnet50_model)
criterion = torch.nn.CosineEmbeddingLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001) # use Adam optimizer instead of Adagrad

# %%
train_model(model, 75, visualize=True)

# %% [markdown]
# - We see that the training loss is decreasing and the moving average of our validation accuracy is increasing, thus we can say that our model is learning

# %%
torch.save(model, 'epochs75_aug_reg_adam.pt')

# %%
model = torch.load('epochs75_aug_reg_adam.pt')

# %% [markdown]
# ### Cosine Similarity Visualization

# %%
dataiter = iter(test_dataloader)
for i in range(3):
    img0, img1, label = next(dataiter)
    concatenated = torch.cat((img0, img1),0)

    output1, output2 = model(img0.cuda(), img1.cuda())
    cosine_similarity = torch.nn.functional.cosine_similarity(output1, output2)
    print(cosine_similarity.cpu().detach().numpy().T)
    img = torchvision.utils.make_grid(concatenated)
    npimg = img.numpy()
    plt.figure(figsize = (16, 8))
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# %% [markdown]
# As we can see that the cosine similarities of the embeddings of the similar images are close to 1 and the cosine similarities of the embeddings of the dissimilar images further away from 1 closer to 0 and -1

# %% [markdown]
# ### Test Accuracy

# %%
threshold = 0.8
correct = 0
for i, data in enumerate(test_dataloader):
  img0, img1, label = data

  if torch.cuda.is_available():
    output1, output2 = model(img0.cuda(), img1.cuda())
  else:
    output1, output2 = model(img0, img1)

  cosine_similarity = torch.nn.functional.cosine_similarity(output1, output2)
  pred_label = cosine_similarity.cpu().detach().numpy()
  pred_label[pred_label > threshold] = 1.0
  pred_label[pred_label <= threshold] = -1.0

  correct += np.sum(pred_label == label.cpu().detach().numpy())

print("Accuracy:", correct*100/(8*len(test_dataloader)))

# %% [markdown]
# - We see that the test accuracy of the model with Adam optimizer with appropriately changed learning rate performs as well as the model with Adagrad optimizer.

# %% [markdown]
# ## Q9: Testing on the Test Split

# %% [markdown]
# So the model which performed best so far was the one with constant learning rate with image augmentation and regularization with Adagrad optimizer, so we will use this model for the final testing on a new test split. We will choose the threshold for cosine similarity as 0.8, if the cosine similarity is greater than 0.8 we will consider the images to be similar, else we will consider them to be dissimilar.

# %%
test_split = SiameseNetworkDataset(personNames=test_data_names, dataset_size=500, transform=transforms.Compose([ # generate a larger test set of 500 images
                                     transforms.Resize((224, 224)), transforms.ToTensor()]))
test_split_dataloader = DataLoader(test_split, shuffle=False, batch_size=4)

# %%
model = torch.load('epochs75_reg.pt') # load best model so far

# %%
dataiter = iter(test_split_dataloader)
for i in range(3):
    img0, img1, label = next(dataiter)
    concatenated = torch.cat((img0, img1),0)

    output1, output2 = model(img0.cuda(), img1.cuda())
    cosine_similarity = torch.nn.functional.cosine_similarity(output1, output2)
    print(cosine_similarity.cpu().detach().numpy().T)
    img = torchvision.utils.make_grid(concatenated, nrow=4)
    npimg = img.numpy()
    plt.figure(figsize = (16, 8))
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# %%
threshold = 0.8
correct = 0
for i, data in enumerate(test_dataloader):
  img0, img1, label = data

  if torch.cuda.is_available():
    output1, output2 = model(img0.cuda(), img1.cuda())
  else:
    output1, output2 = model(img0, img1)

  cosine_similarity = torch.nn.functional.cosine_similarity(output1, output2)
  pred_label = cosine_similarity.cpu().detach().numpy()
  pred_label[pred_label > threshold] = 1.0
  pred_label[pred_label <= threshold] = -1.0

  correct += np.sum(pred_label == label.cpu().detach().numpy())

print("Accuracy:", correct*100/(8*len(test_dataloader)))

# %% [markdown]
# ### Thus we see that the model is able to achieve a test accuracy of 0.807 on the new test split, which is very good given we have trained for just 75 epochs.

# %% [markdown]
# ## Q10: Testing the model on my and my friends' images and inference

# %%
friends = ['rohan', 'ayushman', 'karthikeyan'] # list of friends

# %%
root_directory = 'friends'

def get_rohan():
    # get a random image of rohan
    k = os.listdir('friends/rohan')
    img0 = PIL.Image.open(root_directory + f'/rohan' + f'/{random.choice(k)}')
    img0 = img0.convert("RGB")
    img0 = transforms.Resize((224, 224))(img0)
    img0 = transforms.ToTensor()(img0)
    img0 = torch.reshape(img0, (1, 3, 224, 224))
    return img0

def get_ayushman():
    # get a random image of ayushman
    k = os.listdir('friends/ayushman')
    img0 = PIL.Image.open(root_directory + f'/ayushman' + f'/{random.choice(k)}')
    img0 = img0.convert("RGB")
    img0 = transforms.Resize((224, 224))(img0)
    img0 = transforms.ToTensor()(img0)
    img0 = torch.reshape(img0, (1, 3, 224, 224))
    return img0

def get_karthikeyan():
    # get a random image of karthikeyan
    k = os.listdir('friends/karthikeyan')
    img0 = PIL.Image.open(root_directory + f'/karthikeyan' + f'/{random.choice(k)}')
    img0 = img0.convert("RGB")
    img0 = transforms.Resize((224, 224))(img0)
    img0 = transforms.ToTensor()(img0)
    img0 = torch.reshape(img0, (1, 3, 224, 224))
    return img0

def test_with_friends(f, g):
    img0 = f() # get a random image of friend 1
    img1 = g() # get a random image of friend 2
    output1, output2 = model(img0.cuda(), img1.cuda()) # forward pass
    cosine_similarity = torch.nn.functional.cosine_similarity(output1, output2)
    print(cosine_similarity.cpu().detach().numpy().T) # print cosine similarity
    plt.figure(figsize = (16, 8))
    plt.title("Similarity Score : " + str(cosine_similarity.cpu().detach().numpy().T))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(np.rot90(img0.numpy().T.reshape(224, 224, 3), axes=(1,0)))
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(np.rot90(img1.numpy().T.reshape(224, 224, 3), axes=(1,0)))
    plt.show()

# %%
test_with_friends(get_rohan, get_ayushman)

# %%
test_with_friends(get_ayushman, get_karthikeyan)

# %%
test_with_friends(get_rohan, get_karthikeyan)

# %%
test_with_friends(get_rohan, get_rohan)

# %%
test_with_friends(get_ayushman, get_ayushman)

# %%
test_with_friends(get_karthikeyan, get_karthikeyan)

# %% [markdown]
# - The model performs well on differentiating between images of me and my friends as well, however the threshold of 0.8 is not able to differentiate between images, this is because all of these images were taken in same lighting and background conditions (Tinkerer's Lab, IIT Bombay), however for a finer threshold such as 0.9 or 0.95 the model does a better job of differentiating between images of me and my friends.

# %% [markdown]
# # Part B

# %% [markdown]
# ## Q11 : A generative model for generating face images using GAN (Discriminator + Generator)
# 
# - Reference used : https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html for the DCGAN architecture from official PyTorch tutorials

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import torchvision
import matplotlib.pyplot as plt

# %%
root_directory = "lfw" # root directory of dataset
batch_size = 64 # batch size
image_size = 64 # image size
z_dimensions = 100 # latent space dimensions
num_gen_features = 100 # number of features in generator
num_disc_features = 100
num_epochs = 10 # number of epochs
lr = 0.0002 # learning rate
beta_reg = (0.5, 0.999) # beta for Adam optimizer

# %%
def visualize_images(title, img_np):
    # visualize images
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(img_np,(1,2,0)))
    plt.show()

# %%
dataset = torchvision.datasets.ImageFolder(root=root_directory,
                           transform=transforms.Compose([
                               transforms.Resize(image_size), # resize to 64x64
                               transforms.CenterCrop(image_size), # crop to 64x64
                               transforms.ToTensor(), # convert to tensor
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # normalize
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

sample_images = next(iter(dataloader))
# visualize a batch of images
visualize_images("Sample Real Images", torchvision.utils.make_grid(sample_images[0].to(device), padding=2, normalize=True).cpu())


# %%
class Generator(nn.Module): # generator class
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d( # input is Z, going into a convolution
                z_dimensions, num_gen_features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_gen_features * 8), # batch normalization
            nn.ReLU(True), # ReLU activation
            nn.ConvTranspose2d(num_gen_features * 8, # transpose convolution
                               num_gen_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_gen_features * 4), # batch normalization
            nn.ReLU(True), # ReLU activation
            nn.ConvTranspose2d(num_gen_features * 4, # transpose convolution
                               num_gen_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_gen_features * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_gen_features * 2, # transpose convolution
                               num_gen_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_gen_features),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_gen_features, 3, 4, 2, 1, bias=False), # transpose convolution
            nn.Tanh() # Tanh activation
        )

    def forward(self, input):
        return self.main(input)

# %%
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, num_disc_features, 4, 2, 1, bias=False), # convolution
            nn.LeakyReLU(0.2, inplace=True), # LeakyReLU activation
            nn.Conv2d(num_disc_features, num_disc_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_disc_features * 2, num_disc_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_disc_features * 4, num_disc_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_disc_features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid() # sigmoid activation
        )

    def forward(self, input):
        return self.main(input)

# %%
disc_model = Discriminator().to(device)
gen_model = Generator().to(device)
criterion = nn.BCELoss() # binary cross entropy loss
fixed_noise = torch.randn(64, z_dimensions, 1, 1, device=device) # fixed noise for visualization

# %%
optimizer_disc = optim.Adam(disc_model.parameters(), lr=lr, betas=beta_reg) # Adam optimizer
optimizer_gen = optim.Adam(gen_model.parameters(), lr=lr, betas=beta_reg)

actual_img_label = 1.
fake_img_label = 0.

# %%
img_list = []
iters = 0

for epoch in range(num_epochs): # training loop

    avg_d_error = [] # average discriminator error
    avg_g_error = [] # average generator error
    avg_d_x = [] # average D(x)
    avg_g_d_z = [] # average D(G(z))
    with torch.no_grad():
            # visualize fake images
            fake_image = gen_model(fixed_noise).detach().cpu()
    img_list.append(torchvision.utils.make_grid(fake_image, padding=2, normalize=True)) # append fake images to list
    for i, data in enumerate(dataloader, 0):
        disc_model.zero_grad() # zero out gradients
        real_image = data[0].to(device) # real images
        b_size = real_image.size(0) # batch size
        label = torch.full((b_size,), actual_img_label, dtype=torch.float, device=device)
        output = disc_model(real_image).view(-1)
        disc_error = criterion(output, label)
        disc_error.backward() # backpropagate
        D_x = output.mean().item() # D(x)

        noise = torch.randn(b_size, z_dimensions, 1, 1, device=device) # noise
        fake = gen_model(noise) # fake images
        label.fill_(fake_img_label) # fake labels
        output = disc_model(fake.detach()).view(-1) # detach to avoid backpropagating through generator
        disc_error_fake = criterion(output, label) # discriminator error on fake images
        disc_error_fake.backward() # backpropagate
        D_G_z1 = output.mean().item() # D(G(z))
        disc_error_total = disc_error_fake + disc_error # total discriminator error
        optimizer_disc.step() # update discriminator parameters

        gen_model.zero_grad() # zero out gradients
        label.fill_(actual_img_label) # real labels
        output = disc_model(fake).view(-1) # discriminator output on fake images
        gen_error = criterion(output, label) # generator error
        gen_error.backward() # backpropagate
        D_G_z2 = output.mean().item() # D(G(z))
        optimizer_gen.step() # update generator parameters 

        avg_d_error.append(disc_error_total.item())
        avg_g_error.append(gen_error.item())
        avg_d_x.append(D_x)
        avg_g_d_z.append((D_G_z1 + D_G_z2)/2) # average D(G(z))
        iters += 1
    print("Epoch number {} : Avg D(x) {} : Avg D(G(z)) {} : Avg D Loss {} : Avg G Loss {}".format( # print statistics
        epoch, np.mean(avg_d_x), np.mean(avg_g_d_z), np.mean(avg_d_error), np.mean(avg_g_error)))

# %%
for i, j in enumerate(img_list):
    visualize_images("Generated Images after {} Epochs".format(i+1), j)

# %%
torch.save(gen_model, 'gan.pt')
torch.save(disc_model, 'gan_disc.pt')

# %%
actual_images = next(iter(dataloader))

visualize_images("Real Images", torchvision.utils.make_grid(actual_images[0].to(device)[:batch_size], padding=5, normalize=True).cpu())
visualize_images("Generated Images", img_list[-1])

# %% [markdown]
# - Thus we see that the discriminator is able to differentiate between real and fake images, and the generator is able to generate images which look like real images.

# %% [markdown]
# ## Q12 : Modification of the GAN to become a conditional GAN (BONUS)

# %% [markdown]
# ### Let us first restrict our dataset to only 5 people and the concept will scale to more people, let us make a GAN capable of generating images of these 5 people with an additional image of the person as input

# %%
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import PIL
import torchvision
import matplotlib.pyplot as plt

# %%
import os
import random

root_directory = "lfw" # root directory of dataset
people = ['Yashwant_Sinha', 'Vince_Carter', 'Vojislav_Kostunica', 'Zinedine_Zidane', 'Tim_Robbins'] # list of people
# if we are able to show that the model can distinguish between these 5 people, we can say that the concept will extend to other people as well

# %%
!mkdir conditional # create a directory to store images of people

# %%
# copy images of people to conditional directory
!cp -r lfw/Yashwant_Sinha conditional
!cp -r lfw/Vince_Carter conditional
!cp -r lfw/Vojislav_Kostunica conditional
!cp -r lfw/Zinedine_Zidane conditional
!cp -r lfw/Tim_Robbins conditional 

# %% [markdown]
# ### We modify the Discriminator to take in an label as input, this image is concatenated with the image to be classified, also the Generator takes in a label as input, this image is concatenated with the noise vector to be fed to the Generator
# 
# - Reference Followed for the Conditional GAN (Video Provided in PS): https://www.youtube.com/watch?v=Hp-jWm2SzR8

# %%
num_classes = len(people) # number of classes
gen_embedding = 100 # embedding size for generator
root_directory = "conditional"

# other hyperparameters same as before in GAN
batch_size = 8
image_size = 64
z_dimensions = 100
num_gen_features = 100
num_disc_features = 100
num_epochs = 5
lr = 0.0002
beta_reg = (0.5, 0.999)

# %% [markdown]
# ### Modified Discriminator, Rest of the architecture remains the same

# %%
class Discriminator(nn.Module):
    def __init__(self, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size # image size
        self.main = nn.Sequential(
            nn.Conv2d(3 + 1, num_disc_features, 4, 2, 1, bias=False), # include label as an additional channel
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_disc_features, num_disc_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_disc_features * 2, num_disc_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_disc_features * 4, num_disc_features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_disc_features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_disc_features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.embed = nn.Embedding(num_classes, img_size*img_size) # embedding layer

    def forward(self, input, labels):
        embedding = self.embed(labels).view(labels.shape[0], 1, self.img_size, self.img_size) # reshape embedding
        input = torch.cat([input, embedding], dim=1) # concatenate embedding with input
        return self.main(input)

# %% [markdown]
# ### Modified Generator, Rest of the architecture remains the same

# %%
class Generator(nn.Module):
    def __init__(self, num_classes, img_size, embed_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.main = nn.Sequential(
            nn.ConvTranspose2d(
                z_dimensions + embed_size, num_gen_features * 8, 4, 1, 0, bias=False), # include label as an additional channel
            nn.BatchNorm2d(num_gen_features * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_gen_features * 8,
                               num_gen_features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_gen_features * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_gen_features * 4,
                               num_gen_features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_gen_features * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_gen_features * 2,
                               num_gen_features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_gen_features),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_gen_features, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.embed = nn.Embedding(num_classes, embed_size)

    def forward(self, input, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3) # reshape embedding
        input = torch.cat([input, embedding], dim=1) # concatenate embedding with input
        return self.main(input)

# %%
def visualize_images(title, img_np):
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title(title)
    plt.imshow(np.transpose(img_np,(1,2,0)))
    plt.show()

# %% [markdown]
# ### Modified Dataset Class to return the label as well

# %%
class CGANDataset(Dataset):
    def __init__(self, root_dir, transform=None, dataset_size=10000):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_size = dataset_size # dataset size
        self.dataset = torchvision.datasets.ImageFolder(root=root_directory, # load dataset
                                                        transform=transforms.Compose([
                                                            transforms.Resize(
                                                                image_size),
                                                            transforms.CenterCrop(
                                                                image_size),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(
                                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                        ]))
        self.map = self.dataset.class_to_idx # map of class to index

    def __getitem__(self, index):
        person = random.choice(os.listdir(root_directory)) # choose a random person

        img_name = random.choice(os.listdir(root_directory + f'/{person}')) # choose a random image of that person

        img = PIL.Image.open(root_directory + f'/{person}' + f'/{img_name}')
        img = img.convert("RGB")
        label = self.map[person] # get label
        if self.transform is not None:
            img = self.transform(img)
        return img, label # return image and label

    def __len__(self):
        return self.dataset_size

# %%
dataset = CGANDataset(root_dir = root_directory, transform=transforms.Compose([
                                                            transforms.Resize(
                                                                image_size),
                                                            transforms.CenterCrop(
                                                                image_size),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(
                                                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                        ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

sample_images = next(iter(dataloader))

visualize_images("Sample Real Images", torchvision.utils.make_grid(sample_images[0].to(device), padding=2, normalize=True).cpu()) # visualize a batch of images
print(sample_images[1])


# %% [markdown]
# ### The labels are also provided to the Discriminator and Generator also with the images

# %%
person_labels = dataset.map # map of class to index
print(person_labels)

# %% [markdown]
# - For simplicity we will restrict to ten classes of people and the concept will extend

# %%
disc_model = Discriminator(num_classes, image_size).to(device) # discriminator model
gen_model = Generator(num_classes, image_size, gen_embedding).to(device) # generator model
criterion = nn.BCELoss() # binary cross entropy loss
fixed_noise = torch.randn(64, z_dimensions, 1, 1, device=device) # fixed noise for visualization
fixed_label = torch.randint(0, num_classes, (64,), device=device) # fixed label examples for visualization

# %%
optimizer_disc = optim.Adam(disc_model.parameters(), lr=lr, betas=beta_reg)
optimizer_gen = optim.Adam(gen_model.parameters(), lr=lr, betas=beta_reg)

actual_img_label = 1.
fake_img_label = 0.

# %%
print("the fixed labels for testing, generated randomly :", fixed_label) # print fixed label examples

# %% [markdown]
# ### Modified Training Loop

# %%
img_list = []
iters = 0

for epoch in range(num_epochs):
    avg_d_error = []
    avg_g_error = []
    avg_d_x = []
    avg_g_d_z = []

    with torch.no_grad():
            fake_image = gen_model(fixed_noise, fixed_label).detach().cpu() # visualize fake images, add fixed label
    img_list.append(torchvision.utils.make_grid(fake_image, padding=2, normalize=True))
    
    for i, (data, labels) in enumerate(dataloader): # get labels from dataloader batch
        disc_model.zero_grad()
        real_image = data.to(device)
        labels = labels.to(device)
        b_size = real_image.size(0)
        label = torch.full((b_size,), actual_img_label, dtype=torch.float, device=device)
        output = disc_model(real_image, labels).view(-1) # add labels to discriminator
        disc_error = criterion(output, label)
        disc_error.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, z_dimensions, 1, 1, device=device)
        fake = gen_model(noise, labels) # add labels to generator
        label.fill_(fake_img_label)
        output = disc_model(fake.detach(), labels).view(-1) # detach to avoid backpropagating through generator
        disc_error_fake = criterion(output, label)
        disc_error_fake.backward()
        D_G_z1 = output.mean().item()
        disc_error_total = disc_error_fake + disc_error
        optimizer_disc.step()

        gen_model.zero_grad()
        label.fill_(actual_img_label)
        output = disc_model(fake, labels).view(-1) # discriminator output on fake images
        gen_error = criterion(output, label)
        gen_error.backward()
        D_G_z2 = output.mean().item()
        optimizer_gen.step()

        avg_d_error.append(disc_error_total.item())
        avg_g_error.append(gen_error.item())
        avg_d_x.append(D_x)
        avg_g_d_z.append((D_G_z1 + D_G_z2)/2)
        iters += 1
    print("Epoch number {} : Avg D(x) {} : Avg D(G(z)) {} : Avg D Loss {} : Avg G Loss {}".format(
        epoch, np.mean(avg_d_x), np.mean(avg_g_d_z), np.mean(avg_d_error), np.mean(avg_g_error)))

# %%
for i, j in enumerate(img_list):
    visualize_images("Generated Images after {} Epochs".format(i+1), j)

# %% [markdown]
# ### Thus we given the label, the conditional GAN is able to generate images of the person

# %%
torch.save(gen_model, 'cgan_gen.pt')
torch.save(disc_model, 'cgan_disc.pt')

# %% [markdown]
# ## Using the Siamese Network to identify the label of the input image

# %%
model = torch.load('epochs75_reg.pt')
root_directory = 'conditional'

def get_person(person, vis=True):
    # get a random image of a person
    k = random.choice(os.listdir(f'conditional/{person}'))
    img = PIL.Image.open(root_directory + f'/{person}' + f'/{k}')
    img = img.convert("RGB")
    imgt = transforms.Resize((224, 224))(img)
    imgt = transforms.ToTensor()(imgt)
    imgt = torch.reshape(imgt, (1, 3, 224, 224))
    if(vis):
        # visualize image as 64 x 64 which will be generated by the CGAN for comparison
        imgr = transforms.Resize((64, 64))(img)
        imgr = transforms.ToTensor()(imgr)
        imgr = torch.reshape(imgr, (1, 3, 64, 64))
        visualize_images("Image", torchvision.utils.make_grid(imgr, padding=2, normalize=True))
    return imgt # image tensor to be input to the Siamese Network

def get_best_label(img, siamese_model):
    # get the best label for the image using the Siamese Network
    img0 = img  
    max_similarity = 0 # maximum similarity
    best_label = None # best label
    for person in people:
        img1 = get_person(person, vis=False)
        output1, output2 = siamese_model(img0.cuda(), img1.cuda())
        cosine_similarity = torch.nn.functional.cosine_similarity(output1, output2) # cosine similarity
        if cosine_similarity > max_similarity: # update maximum similarity and best label
            max_similarity = cosine_similarity
            best_label = person
    return best_label

# %%
print(person_labels)
input_img = get_person('Tim_Robbins') # get a random image of Tim Robbins

# %%
pred_label = get_best_label(input_img, model) # get the best label for the image

# %%
pred_label = person_labels[pred_label]

# %%
pred_label # print the predicted label

# %% [markdown]
# ### Thus our Siamese Network is able to identify the label of the input image, now lets use this label to generate images of the person using the conditional GAN

# %%
fixed_noise = torch.randn(1, z_dimensions, 1, 1, device=device) # fixed noise
fixed_label = torch.tensor([pred_label], device=device) # use the predicted label as the fixed label
fake_image = gen_model(fixed_noise, fixed_label).detach().cpu() # generate image

# %%
visualize_images("Generated Image of Tim Robbins", torchvision.utils.make_grid(fake_image, padding=2, normalize=True)) # visualize image

# %% [markdown]
# ### Thus we see that the conditional GAN is able to generate images of the person given another image of the person, and it makes sure not to generate images of other people or copy the input image


