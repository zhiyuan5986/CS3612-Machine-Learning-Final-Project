from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import models, transforms
import shap
import os
from PIL import Image

class Shapley():
    def __init__(self, model="resnet34", useGradients = True):
        self.useGradients = useGradients
        model_urls = {"resnet34":"https://download.pytorch.org/models/resnet34-b627a593.pth"}
        self.model = None
        self.model_name = None
        if model.lower() == "resnet34":
            self.model = models.resnet34(pretrained = True)
            self.model_name = "resnet34"
        else:
            self.model = models.vgg16(pretrained=True)
            self.model_name = "vgg16"
        self.model.eval()
    
    def fit(self, image_path = "./dataset/cat_dog.jpg", output_labels = [242, 282]):
        image = Image.open(image_path)
        image = transforms.Resize((224, 224))(image)

        if self.useGradients:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            def normalize(image):
                if image.max() > 1:
                    image = image / 255
                image = (image - mean) / std
                # in addition, roll the axis so that they suit pytorch
                return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()
            image = np.array(image).astype(np.float32)[np.newaxis, :] / 255
            self.image = image

            e = None
            if self.model_name == "resnet34":
                e = shap.GradientExplainer((self.model, self.model.layer1[1].conv2), normalize(image))
            else:
                e = shap.GradientExplainer((self.model, self.model.features[7]), normalize(image))
            shap_values,indexes = e.shap_values(normalize(image), ranked_outputs=2, nsamples=200)

            self.shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
            self.class_labels = np.arange(0, 1000)[indexes]
        else:
            image = np.array(image).astype(np.float32)
            masker = shap.maskers.Image("inpaint_telea", image.shape)

            def f(x):
                return self.model.forward(torch.from_numpy(x).permute(0,3,1,2).type(torch.float)).cpu().detach().numpy().astype(np.float32)
            class_labels = list(range(0,1000)) 
            explainer = shap.Explainer(f, masker, output_names=class_labels)

            self.shap_values = explainer(image[np.newaxis, :], max_evals=5000, batch_size=50, outputs=output_labels)

    def visualize(self):
        if self.useGradients:
            shap.image_plot(self.shap_values, self.image, self.class_labels)
            plt.savefig(f"./output/Task3_5/{self.model_name}_visualize_gradients.png", bbox_inches="tight")
        else:
            shap_values = self.shap_values
            shap_values.data = shap_values.data.astype(np.uint8)
            shap.image_plot(shap_values, show=False)
            plt.savefig(f"./output/Task3_5/{self.model_name}_visualize_original_image.png", bbox_inches="tight")

if __name__ == "__main__":
    if not os.path.exists("./output/"):
        os.mkdir("./output/")
    if not os.path.exists("./output/Task3_5/"):
        os.mkdir("./output/Task3_5/")

    for useGradients in [True, False]:
        for model_name in ["resnet34", "vgg16"]:
            shapley = Shapley(model=model_name, useGradients=useGradients)
            shapley.fit(image_path="./dataset/cat_dog.jpg", output_labels=[242, 282])
            shapley.visualize()