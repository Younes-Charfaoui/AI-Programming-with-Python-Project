# other libraries
from PIL import Image
import numpy as np
import argparse
import json

# torch libraries
import torchvision.transforms as transforms
from torchvision import models
import torch
from torchvision import datasets, transforms, models


# function for getting the arguments.
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('image', type=str)

    parser.add_argument('checkpoint', type=str)

    parser.add_argument('--gpu', action='store_true')

    parser.add_argument('--topk', type=int, default=5)

    parser.add_argument('--category_names', type=str, default=' ')

    args = parser.parse_args()
    return args

# loading the saved model after training.
def load_checkpoint(checkpoint):
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model.features = checkpoint['features']
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

# preprocess the image
def process_image(image):
    im = Image.open(image)
    im = im.resize((256, 256))
    cropped_im = im.crop((0, 0, 249, 249))
    im_to_np = np.array(cropped_im)
    im_to_np = (im_to_np / 45) - np.array([0.485, 0.456, 0.406]) / np.array([0.229, 0.224, 0.225])
    im_to_np = im_to_np.transpose(2, 0, 1)
    return torch.from_numpy(im_to_np)


# predict the category of the image
def predict(image, device, model, topk):


    if topk > len(model.class_to_idx):
        topk = len(model.class_to_idx)
    model.to(device)
    with torch.no_grad():
        model.eval()
        image = image.float().to(device)
        output = model.forward(image)
        prediction = torch.exp(output).data[0].topk(topk)
        return prediction

# the main function to launch the whole process
def main():

    #getting the arguments
    args = get_args()
    if args.gpu:
        device = 'cuda:0'
    else:
        device = 'cpu'

    # loading the trained model
    model = load_checkpoint(args.checkpoint)

    # loading the image and process it.
    image = process_image(args.image).unsqueeze_(0)

    probs, classes = predict(image, device, model, args.topk)
    probs, classes = probs.cpu().numpy(), classes.cpu().numpy()

    # printing the results
    if args.category_names != ' ':
        indexes = []
        names = []

        # loading the json file of the category names
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)

        for category in classes:
            indexes.append(model.class_to_idx[str(category + 1)])
        for name in indexes:
            names.append(cat_to_name[str(name + 1)])
        for result in range(len(probs)):
            print('Rank: {:<3}, Name: {}, Class: {:<3}, Prob: {}\n'.format(result + 1, names[result], classes[result], probs[result]))
            
    else:
        for result in range(len(probs)):
            print('Rank: {:<3}, Class: {:<3}, Prob: {:.4f}\n'.format(result + 1, classes[result], probs[result]))

# calling the main function
if __name__ == "__main__":
    main()