import json
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import base64
import io
from PIL import Image as PILImage
import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet


app = FastAPI()

class Image(BaseModel):
    image: str

class EfficientLite(torch.nn.Module):
    def __init__(self):
        super(EfficientLite, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=8)

    def forward(self, x):
        return self.model(x)

@app.post("/upload/")
async def test(imageInfo: Image):
    try:
        image_data = base64.b64decode(imageInfo.image)
        
        img = PILImage.open(io.BytesIO(image_data))

        img.save("trash.jpg")

        return "trash"
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Error in upload")
    
@app.post("/") 
async def predictCategory(imageInfo: Image):
    try:
        image_data = base64.b64decode(imageInfo.image)
        
        img = PILImage.open(io.BytesIO(image_data))

        img.save("trash.jpg")

        predict(img)

        return {"message": "Image received and saved successfully"}
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Error in upload")


def predict():
    try:
        img = Image.open("trash.jpg")
        print("Image opened successfully.")
    except Exception as e:
        print("Error opening image:", e)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    input_image = transform(img).unsqueeze(0)
    print("transformed")

    model = EfficientLite()
    model.load_state_dict(torch.load('model_state_dict.pth'))
    model.eval()

    with torch.no_grad():
        output = model(input_image)

    _, predicted = torch.max(output, 1)

    return predicted.item()


