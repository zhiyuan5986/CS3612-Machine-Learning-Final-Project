from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import models, transforms
import shap
import os
from PIL import Image

class Shapley():
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
    
    def fit(self, image_path = "./dataset/cat_dog.jpg", output_labels = [242, 282]):
        image = Image.open(image_path)

        trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        image = trans(image)

        cat_dog = image.permute(1,2,0).numpy().astype(np.float32)

        self.model.eval()

        # logits = self.model.forward(torch.from_numpy(cat_dog[np.newaxis, :]).permute(0,3,1,2).type(torch.float).to(self.device)).cpu().detach().numpy().astype(np.float32)
        # pred = logits.argmax(axis=1)

        masker = shap.maskers.Image("inpaint_telea", cat_dog.shape)

        def f(x):
            return self.model.forward(torch.from_numpy(x).permute(0,3,1,2).type(torch.float).to(self.device)).cpu().detach().numpy().astype(np.float32)
        class_labels = list(range(0,1000)) 
        explainer = shap.Explainer(f, masker, output_names=class_labels)

        self.shap_values = explainer((cat_dog)[np.newaxis, :], max_evals=500, batch_size=50, outputs=output_labels)

    def visualize(self):
        shap.image_plot(self.shap_values, show=False)
        plt.savefig(f"./output/Task3_5/{self.model_name}.png", bbox_inches="tight")

if __name__ == "__main__":
    if not os.path.exists("./output/"):
        os.mkdir("./output/")
    if not os.path.exists("./output/Task3_5/"):
        os.mkdir("./output/Task3_5/")

    # ResNet34
    shapley = Shapley(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), model="resnet34")
    shapley.fit(image_path="./dataset/cat_dog.jpg", output_labels=[242, 282])
    shapley.visualize()

    # VGG-16
    shapley = Shapley(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), model="vgg16")
    shapley.fit(image_path="./dataset/cat_dog.jpg", output_labels=[242, 282])
    shapley.visualize()