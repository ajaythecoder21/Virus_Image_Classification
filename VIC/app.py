# app.py

import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template, url_for
import torchvision.transforms as transforms
from PIL import Image
import io
app = Flask(__name__)


import os
#os.environ['FLASK_RUN_PORT']='5001'
os.environ['FLASK_DEBUG'] = 'True'

class VirusCNNModelV0(nn.Module):
  def __init__(self, input_features, hidden_features, output_features):
    super().__init__()
    self.conv_layer_1 = nn.Sequential(
        nn.Conv2d(in_channels = input_features, out_channels=hidden_features, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_features, out_channels=hidden_features,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )
    self.conv_layer_2 = nn.Sequential(nn.Conv2d(in_channels = hidden_features, out_channels=hidden_features, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_features, out_channels=hidden_features,kernel_size=3,stride=1,padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_features, out_channels=hidden_features,kernel_size=3,stride=1,padding=1),
        nn.MaxPool2d(kernel_size=2,stride=2))
    self.classifier_layer = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_features*16*16, out_features=output_features)
    )
  def forward(self, x):
    x = self.conv_layer_1(x)
    #print(f"Shape after first Layer: {x.shape}")
    x = self.conv_layer_2(x)
    #print(f"Shape after second Layer: {x.shape}")
    x = self.classifier_layer(x)
    return x

model = VirusCNNModelV0(3, 10, 3)
device = torch.device("cpu")
model.load_state_dict(torch.load("virus_model.pt"))
model.eval()

transform = transforms.Compose([
   transforms.Resize((64,64)),
   transforms.ToTensor()
])

@app.route("/")
def home():
   return render_template('index.html', home_url=url_for('home'))

@app.route("/about")
def about():
  return render_template('about.html', about_url=url_for('about'))
@app.route("/", methods=["GET", "POST"])
def predict():
   class_names = ['Covid', 'Normal', 'Viral Pneumonia']
   if request.method == "POST":
    data = request.files['image'].read()
    image = transform(Image.open(io.BytesIO(data)).convert("RGB")).unsqueeze(0)
    print(image)
    output = torch.softmax(model(image), dim=1)
    pred_labels_prob = {class_names[i]: float(output[0][i]) for i in range(len(class_names))}
    max_value = (max(pred_labels_prob.values()))
    max_key = [k for k, v in pred_labels_prob.items() if v == max_value]
    print(max_key[0])

    #return jsonify({"output_ratios": pred_labels_prob, "result": max_key[0]})
    context_dict = {"output_ratios": pred_labels_prob, "result": max_key[0]}
    return render_template('index.html', **context_dict)
if __name__ == "__main__":

    app.run(port=8080)
