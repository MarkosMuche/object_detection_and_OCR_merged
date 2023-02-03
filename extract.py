

# This one is to be run after training, only for prediction

from donut import DonutModel
from PIL import Image
import torch
# change the follwoing line after training

model_path = "/content/drive/MyDrive/Ben_mery/models/result/train_ben_receipt/20221130_203401"
#"/content/donut/result/train_ben/20220918_160140"
model = DonutModel.from_pretrained(model_path)
if torch.cuda.is_available():
    model.half()
    device = torch.device("cuda")
    model.to(device)
else:
    model.encoder.to(torch.bfloat16)
model.eval()
#/content/donut/dataset/ben-receipt/train/1021-receipt.jpg
#/content/drive/MyDrive/old_laptop/reciepts/reciept.png
image_path = "/content/donut/dataset/ben-receipt/train/1024-receipt.jpg"  
image = Image.open(image_path).convert("RGB")
output = model.inference(image=image, prompt="<s_ben-receipt>")
output