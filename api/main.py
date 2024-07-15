# Importing necessary libraries
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from notebook.network import Net
import torch
from torchvision import transforms
from PIL import Image
import io

app = FastAPI()

# Load the PyTorch model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(num_classes=28).to(device)

# Load the model state_dict
model.load_state_dict(torch.load('notebook/nn_model/best_model.pth', map_location=device))

# Set the model to evaluation mode
model.eval()

# Define the transformations for input images
test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# List of class names
class_names = [
    'Apple__Healthy', 'Apple__Rotten', 'Banana__Healthy', 'Banana__Rotten',
    'Bellpepper__Healthy', 'Bellpepper__Rotten', 'Carrot__Healthy', 'Carrot__Rotten',
    'Cucumber__Healthy', 'Cucumber__Rotten', 'Grape__Healthy', 'Grape__Rotten',
    'Guava__Healthy', 'Guava__Rotten', 'Jujube__Healthy', 'Jujube__Rotten',
    'Mango__Healthy', 'Mango__Rotten', 'Orange__Healthy', 'Orange__Rotten',
    'Pomegranate__Healthy', 'Pomegranate__Rotten', 'Potato__Healthy', 'Potato__Rotten',
    'Strawberry__Healthy', 'Strawberry__Rotten', 'Tomato__Healthy', 'Tomato__Rotten'
]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Apply test transformations
        image_tensor = test_transforms(image).unsqueeze(0)  # Add a batch dimension
        
        # Move the tensor to the same device as the model
        image_tensor = image_tensor.to(device)
        
        # Make a prediction with the model
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
        
        # Convert the prediction to a readable format
        class_index = predicted.item()
        predicted_class = class_names[class_index]
        
        return {"predicted_class": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
