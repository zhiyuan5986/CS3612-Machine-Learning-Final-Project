from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import os
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from matplotlib import colormaps


class GradCAM():
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

    def fit(self, image_path = "./dataset/cat_dog.jpg", label = 282):
        image = Image.open(image_path)
        trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        image = trans(image)
        self.cat_dog = image.permute(1,2,0)
        self.label = label

        self.model.eval()
        features_out_hook = []
        gradients_out_hook = []

        def hook_forward(module, fea_in, fea_out):
            features_out_hook.append(fea_out)
            return None
        def hook_backward(module, grad_in, grad_out):
            gradients_out_hook.append(grad_out)
            return None
        
        hf = None
        hb = None
        if self.model_name == "resnet34":
            hf = self.model.layer4[2].conv1.register_forward_hook(hook=hook_forward)
            hb = self.model.layer4[2].conv1.register_backward_hook(hook=hook_backward)
        else:
            hf = self.model.features[-1].register_forward_hook(hook=hook_forward)
            hb = self.model.features[-1].register_backward_hook(hook=hook_backward)


        logits = self.model.forward(self.cat_dog.permute([2,0,1]).unsqueeze(dim=0).type(torch.float).to(self.device))
        hf.remove()

        loss = logits[0,self.label]
        loss.backward()
        hb.remove()

        alpha_cat = gradients_out_hook[0][0].mean(dim=[2,3], keepdim=True).squeeze(dim=0)
        L = F.relu(alpha_cat.mul(features_out_hook[0].squeeze(dim=0)).sum(dim=0))
        L /= torch.max(L)
        self.L = L

    def visualize(self):

        fig, ax = plt.subplots()
        ax.axis('off')

        ax.imshow(to_pil_image(self.cat_dog.permute([2,0,1]), mode='RGB'))

        overlay = to_pil_image(self.L.detach().cpu(), mode='F').resize((224, 224), resample=Image.BICUBIC)

        cmap = colormaps['jet']
        overlay = (255*cmap(np.asarray(overlay) ** 2)[:,:,:3]).astype(np.uint8)

        ax.imshow(overlay, alpha=0.4, interpolation="nearest")

        plt.title(f"Label = {self.label}")

        plt.savefig(f"./output/Task3_4/{self.model_name}_label{self.label}.png", bbox_inches = "tight")

if __name__ == "__main__":
    if not os.path.exists("./output/"):
        os.mkdir("./output/")
    if not os.path.exists("./output/Task3_4/"):
        os.mkdir("./output/Task3_4/")

    # ResNet34
    gradcam = GradCAM(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), model="resnet34")

        # cat
    gradcam.fit(label=282)
    gradcam.visualize()

        # dog
    gradcam.fit(label=242)
    gradcam.visualize()

    # VGG-16
    gradcam = GradCAM(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), model="vgg16")

        # cat
    gradcam.fit(label=282)
    gradcam.visualize()

        # dog
    gradcam.fit(label=242)
    gradcam.visualize()
