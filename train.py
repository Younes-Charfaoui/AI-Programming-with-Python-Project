# importing the libraries for training
import torchvision.transforms as transforms
import numpy as np
import torch
from torch import nn
from torch import optim
from collections import OrderedDict
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# other libraries.
import argparse
import os
import sys
from workspace_utils import active_session

# function for adding the arguments.
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str)

    parser.add_argument('--save_dir', type=str, default='checkpoints')

    parser.add_argument('--gpu',  action='store_true')

    parser.add_argument('--epochs',  type=int, default=6)

    parser.add_argument('--learning_rate',  type=float, default=0.001)
    
    parser.add_argument('--arch',  type=str, default='densenet')

    parser.add_argument('--hidden_units', type=int, default=256)

    args = parser.parse_args()
    return args


# defining the directories of our data.
def set_directory_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data = {'train': train_dir, 'valid': valid_dir, 'test': test_dir}
    return data


def set_loaders(data):

    # creating the transformation
    data_transforms_train = transforms.Compose([
        transforms.RandomRotation(40),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data_transforms_valid = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_transforms_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # creating the image folders
    train_data = datasets.ImageFolder(data['train'], data_transforms_train)
    valid_data = datasets.ImageFolder(data['valid'], data_transforms_valid)
    test_data = datasets.ImageFolder(data['test'], data_transforms_test)

    # creating the data loaders
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = DataLoader(valid_data, batch_size=64, shuffle=True)
    testloader = DataLoader(test_data, batch_size=64, shuffle=True)

    loaders = {'train': trainloader, 'valid': validationloader, 'test': testloader, 'trainset': train_data}

    return loaders

# loading the pretrained network
def load_network():
    densenet161 = models.densenet161(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    vgg16 = models.vgg16(pretrained=True)

    select_models = {'densenet': densenet161, 'alexnet': alexnet, 'vgg': vgg16}
    if arch in select_models:
        in_arch = arch
    else:
        in_arch = 'densenet'
    model = select_models[in_arch]
    
    return model, in_arch, select_models

# building the classifier
def build_classifier(model, hidden):

    # freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier_input_size = model.classifier.in_features
    output_size = 102

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(classifier_input_size, hidden)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden, output_size)),
        ('out', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    return model


def validation(model, validloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    valid_loss = 0
    accuracy = 0
    for inputs, labels in validloader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1]).to(device)
        accuracy += equality.type(torch.cuda.FloatTensor).mean()

    return valid_loss, accuracy


def train(model, device, epochs, trainloader, validationloader, learning_rate):
    print('Training..', flush = True)
    # initialization
    model.train()
    steps = 0
    criterion = nn.CrossEntropyLoss()
    print_every = 30
    model.to(device)
    for i in range(epochs):
        model.train()
        running_loss = 0
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # back to evaluation mode
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validationloader, device)

                print("Epoch: {}/{}... ".format(i + 1, epochs),
                      "Training Loss: {}.. ".format(running_loss / print_every),
                      "validation Loss: {}.. ".format(valid_loss / len(validationloader)),
                      "validation Accuracy: {}".format(accuracy / len(validationloader)), flush = True)

                running_loss = 0

                # back to train mode
                model.train()


# function to test the model.
def test(model, testloader, device):
    print('Performing the validation on the test set', flush = True)
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for inputs, labels in testloader:
            model.eval()
            total += len(labels)
            inputs, label = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            predictions = torch.exp(output).max(dim=1)[1]
            for i in range(len(predictions)):
                if predictions[i] == label[i]:
                    correct += 1
        print('{} Correct out of {}, {:.2f}% Accurate'.format(correct, total, (correct / total) * 100))


# function to save the model
def save_model(model, arch, epochs, path, trainset, select_models):
    print('Saving the model to ./{}/checkpoint.pth'.format(path), flush = True)
    model.class_to_idx = trainset.class_to_idx
    checkpoint = {
        'model': select_models[arch],
        'features': model.features,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'epochs': epochs
    }
    if not os.path.exists(path):
        print('save directories...', flush = True)
        os.makedirs(path)
    torch.save(checkpoint, path + '/checkpoint.pth')


# the main function to mix all the previous functions
def main():
    # getting the arguments
    args = get_args()

    # chosing the GPU type
    if args.gpu:
        device = 'cuda:0'
    else:
        device = 'cpu'

    # creating data directories
    data = set_directory_data(args.data_dir)

    # making the data loaders
    loaders = set_loaders(data)

    # loading the model
    model, in_arch, select_models = load_network(args.arch)

    # building the classifier
    model = build_classifier(model, args.hidden_units)

    # training the model
    train(model, device, args.epochs, loaders['train'], loaders['valid'], args.learning_rate)

    # testing the model
    test(model, loaders['test'], device)

    # saving the Results.
    save_model(model, args.epochs, args.save_dir, loaders['trainset'])

    # complete.
    print('Training Completed Successfully !', flush = True)


if __name__ == "__main__":
    sys.stdout.flush()
    with active_session():
         main()