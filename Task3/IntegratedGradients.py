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
        # print(gradients.shape)

        integrated_gradients = ((gradients + gradient_baseline) / 2) * (inputs - baseline)
        # print(integrated_gradients[:1])

        # return torch.sum(integrated_gradients, dim=(1, 2, 3))
        self.integrated_gradients = torch.sum(integrated_gradients, dim=0)

    def visualize(self):
        # Convert the image and integrated gradients to numpy arrays
        image_np = self.image.permute(1,2,0).detach().cpu().numpy().copy()
        integrated_gradients = self.integrated_gradients.detach().cpu().numpy()

        # Compute the absolute values of the integrated gradients
        # print(np.linalg.norm(integrated_gradients))
        integrated_gradients /= np.linalg.norm(integrated_gradients)
        magnitudes = np.abs(integrated_gradients.mean(axis=0))
        # print(integrated_gradients)

        max_magnitude = magnitudes.max()
        # Compute a threshold for small magnitudes
        threshold = 0.1 * max_magnitude

        # Create a mask for small magnitudes
        mask = magnitudes < threshold
        # print(mask.shape)
        # print(mask)
        
        # image = image.transpose(1,2,0)

        # overlay_np = image_np.copy()
        # overlay_np[mask] = 0
        # overlay_np /= overlay_np.max()

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(image_np)
        axs[0].axis('off')
        axs[0].set_title('Original Image')
        image_np[mask] = 0
        axs[1].imshow(image_np)
        axs[1].axis('off')
        axs[1].set_title('Integrated Gradients with Black Pixels')
        plt.savefig(f"./output/Task3_6/{self.model_name}_label{self.label}.png")
    
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