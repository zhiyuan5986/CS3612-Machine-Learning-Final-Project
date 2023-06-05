from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import models, transforms
import torch.nn.functional as F
import shap
import os
from PIL import Image

class IntegratedGradients():
    def __init__(self, device, model="resnet34"):
        self.device = device
        model_urls = {"resnet34":"https://download.pytorch.org/models/resnet34-b627a593.pth"}
        self.model = None
        self.model_name = None
        if model.lower() == "resnet34":
            self.model = models.resnet34(pretrained = True).to(self.device)
            self.model_name = "resnet34"
        else:
            self.model = models.vgg16(pretrained=True).to(self.device)
            self.model_name = "vgg16"
        self.model.eval()

    def pre_processing(self, image_path = "./dataset/cat_dog.jpg"):
        image = Image.open(image_path)
        trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.image = trans(image)
        self.inputs = self.image.unsqueeze(0).type(torch.float).to(self.device)

    def get_integrated_gradients(self, label = 282, baseline=None, num_steps=100):
        
        def get_gradients(inputs):
            inputs.require_grad = True
            output = F.softmax(self.model(inputs), dim=1)[0, label]
            return torch.autograd.grad(output, inputs)[0]

        inputs = self.inputs
        self.label = label

        if baseline is None:
            baseline = torch.zeros_like(inputs)

        assert inputs.shape == baseline.shape

        inputs.requires_grad = True
        baseline.requires_grad = True

        # Compute the gradient at the baseline input
        gradient_baseline = get_gradients(baseline)

        # Compute the path from baseline to input
        path = torch.linspace(0, 1, num_steps+1).reshape(-1, 1, 1, 1).to(inputs.device)
        path_inputs = baseline + (inputs - baseline) * path # shape: (num_steps, inputs.shape[1:])

        # Compute the gradients along the path
        gradients = []
        for i in range(num_steps+1):
            gradients.append(get_gradients(path_inputs[i,:].unsqueeze(0)))
        gradients = torch.concat(gradients, dim=0)

        integrated_gradients = ((gradients + gradient_baseline) / 2) * (inputs - baseline)

        self.integrated_gradients = torch.sum(integrated_gradients, dim=0)

    def visualize(self):
        image_np = self.image.permute(1,2,0).detach().cpu().numpy()
        integrated_gradients = self.integrated_gradients.detach().cpu().numpy()
        integrated_gradients /= np.linalg.norm(integrated_gradients)
        magnitudes = np.abs(integrated_gradients)
        max_magnitude = magnitudes.max()
        magnitudes /= max_magnitude

        plt.clf()
        plt.axis('off')
        plt.imshow(image_np)
        plt.title("Original Image")
        plt.savefig(f"./output/Task3_6/original_image.png", bbox_inches="tight")

        plt.clf()
        plt.axis('off')
        plt.title(f'Integrated Gradients with Label = {self.label}')
        magnitudes = magnitudes.sum(axis=0)
        magnitudes = np.stack([np.zeros(magnitudes.shape), magnitudes, np.zeros(magnitudes.shape)], axis = 0).transpose(1,2,0)
        plt.imshow(magnitudes)
        plt.savefig(f"./output/Task3_6/{self.model_name}_label{self.label}.png", bbox_inches="tight")


if __name__ == "__main__":
    if not os.path.exists("./output/"):
        os.mkdir("./output/")
    if not os.path.exists("./output/Task3_6/"):
        os.mkdir("./output/Task3_6/")

    # ResNet34
    ig = IntegratedGradients(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), model="resnet34")
    ig.pre_processing(image_path="./dataset/cat_dog.jpg")
    ig.get_integrated_gradients(label=282)
    ig.visualize()
    ig.get_integrated_gradients(label=242)
    ig.visualize()

    # VGG-16
    ig = IntegratedGradients(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), model="vgg16")
    ig.pre_processing(image_path="./dataset/cat_dog.jpg")
    ig.get_integrated_gradients(label=282)
    ig.visualize()
    ig.get_integrated_gradients(label=242)
    ig.visualize()