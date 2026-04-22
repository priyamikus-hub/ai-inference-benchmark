from PIL import Image 
from torchvision import transforms

import csv #handles csv files

import torch #loads pytorch(access to tensors and models)
from torchvision import models #imports pretrained models
import time #measures latency
import torch.nn.functional as F #provides functions for neural network operations
import os #handles file paths
import urllib

import sys # Check Python version

url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
urllib.request.urlretrieve(url, "imagenet_classes.txt")

print("MPS available:", torch.backends.mps.is_available()) #checks if mps is available on mac   

with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

cpu = torch.device("cpu") #run on cpu,base case
mps = torch.device("mps") #run on appl gpu via (metal performance shaders)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) #loads neural network.(ready made)
model.eval() #disables training behaviors for only inference mode

if torch.backends.mps.is_available(): #checks if mps is available on mac and sets device accordingly
    device = torch.device("mps")
    print("Using GPU (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")
model.to(device) #moves models onto cpu mem so model+input are on same device

import os

base_path = os.path.expanduser("~/Desktop/AI Photos")



preprocess = transforms.Compose([ #defines a series of transformations to apply to the input images, including resizing, cropping, and converting to tensor format.
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Get all jpg files in folder
image_files = [f for f in os.listdir(base_path) if f.endswith((".jpg", ".jpeg", ".png", ".avif"))] #lists all files in the specified directory and filters out those that end with ".jpg", creating a list of image file names to be processed.

images = []
valid_files = []

for img_name in image_files:
    img_path = os.path.join(base_path, img_name)
    try: #attempts to open each image file, convert it to RGB format, and apply the defined preprocessing transformations. If successful, the processed image tensor is added to the list of images, and the corresponding file name is added to the list of valid files. If an error occurs (e.g., unsupported file format), a message is printed, and the file is skipped.
        image = Image.open(img_path).convert("RGB")
        input_tensor = preprocess(image)
        images.append(input_tensor)
        valid_files.append(img_name)
    except:
        print(f"Skipping unsupported file: {img_name}")

image_files = valid_files#updates the list of image files to include only those that were successfully processed, ensuring that subsequent steps operate on valid image data.

# Stack into one batch
input_data = torch.stack(images).to(device)

batch_size = 8 #increases batch size to 8 for better GPU utilization, but can be adjusted based on available resources and model requirements
input_data = torch.cat([input_data] * batch_size) #duplicates input data to create a batch of 8 identical images, allowing for more efficient processing on the GPU.


with torch.no_grad(): #nograd diables gradients tracking and runs model once
    _ = model(input_data)

runs = 10 #runs models 10x
times = [] #stores execution time

with torch.no_grad():
    for _ in range(runs):
        start = time.perf_counter() #high precision counter
        output = model(input_data) #runs inference & computational step
        probabilities = F.softmax(output, dim=1) #converts output to probabilities
        top5_prob, top5_catid = torch.topk(probabilities, 5) #gets top 5 predictions
        end = time.perf_counter() #stop timer
        times.append(end - start) #save time length


print("Average latency (ms):", sum(times)/len(times)*1000)

print("Output shape:", output.shape) #prints batch size and classes the model can predict

_, predicted = torch.max(output, 1)
label = labels[predicted[0].item()]

print("\n--- Predictions ---")

for img_idx in range(len(image_files)): #iterates through each image in the batch, retrieves the corresponding file name, prints it, and then displays the top 5 predicted labels along with their confidence scores for that image.
    print(f"\nImage: {image_files[img_idx]}")
    
    probs = top5_prob[img_idx]
    ids = top5_catid[img_idx]
    
    for i in range(5):
        label = labels[ids[i]]
        confidence = probs[i].item() * 100
        print(f"  {i+1}. {label}: {confidence:.2f}%")

with open("predictions.csv", "w", newline="") as file: #opens a new CSV file named "predictions.csv" in write mode, allowing for the storage of the prediction results in a structured format. The newline="" argument ensures that there are no extra blank lines between rows in the CSV file.
    writer = csv.writer(file) #creates a CSV writer object that will be used to write data to the opened CSV file. This object provides methods for writing rows of data in the correct format for CSV files.
    
    # Header
    writer.writerow(["Image", "Rank", "Label", "Confidence (%)"])
    
    # Write data
    for img_idx in range(len(image_files)): #iterates through each image in the batch, retrieves the corresponding file name, and writes the top 5 predicted labels along with their confidence scores for that image into the CSV file. Each row in the CSV file will contain the image name, the rank of the prediction (1 to 5), the predicted label, and the confidence percentage.
        probs = top5_prob[img_idx]
        ids = top5_catid[img_idx]
        
        for i in range(5): #iterates through the top 5 predictions for each image, retrieves the corresponding label and confidence score, and writes this information as a new row in the CSV file using the writer.writerow() method.
            label = labels[ids[i]]
            confidence = probs[i].item() * 100
            writer.writerow([image_files[img_idx], i+1, label, f"{confidence:.2f}"])
