import os
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F

# Global model and device variables
model = None
device = None

# Function to pad the image to the nearest multiple of 32
def pad_image(image, pad_size=32):
    width, height = image.size
    pad_width = (pad_size - width % pad_size) % pad_size
    pad_height = (pad_size - height % pad_size) % pad_size
    padding = (0, 0, pad_width, pad_height)  # Padding on the right and bottom
    return F.pad(ToTensor()(image).unsqueeze(0), padding, mode="constant", value=0)

# Function to perform sliding window prediction on an entire image
def predict_full_image(image_path, patch_size=2048, overlap=0.25, threshold=0.5):
    global model, device

    # Ensure the model is loaded before predicting
    if model is None or device is None:
        load_device_and_model()

    # Load the image
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    
    # Pad the image to be divisible by 32
    padded_image_tensor = pad_image(image, pad_size=32)
    padded_width, padded_height = padded_image_tensor.shape[-1], padded_image_tensor.shape[-2]
    
    # Initialize an empty array for the output
    full_output = np.zeros((padded_height, padded_width), dtype=np.float32)

    # Calculate the stride (with 25% overlap)
    stride = int(patch_size * (1 - overlap))

    # Slide over the image and predict each patch
    model.eval()
    with torch.no_grad():
        for y in range(0, padded_height, stride):
            for x in range(0, padded_width, stride):
                # Extract patch from the padded image
                patch = padded_image_tensor[:, :, y:y+patch_size, x:x+patch_size].to(device)

                # Ensure patch is the correct size by padding if necessary
                if patch.size(2) != patch_size or patch.size(3) != patch_size:
                    pad_right = patch_size - patch.size(3) if patch.size(3) < patch_size else 0
                    pad_bottom = patch_size - patch.size(2) if patch.size(2) < patch_size else 0
                    patch = F.pad(patch, (0, pad_right, 0, pad_bottom))

                # Predict the output for the patch
                output = model(patch)
                output = torch.sigmoid(output).squeeze().cpu().numpy()

                # Apply threshold to get binary prediction
                binary_output = output > threshold

                # Place the predicted patch back into the full output
                y_end = min(y + patch_size, padded_height)
                x_end = min(x + patch_size, padded_width)
                full_output[y:y_end, x:x_end] = binary_output[:(y_end - y), :(x_end - x)]

    # Crop the output back to the original image size (before padding)
    full_output = full_output[:height, :width]

    # Convert the output array to a binary image
    binary_image = Image.fromarray((full_output * 255).astype('uint8'))
    
    # Create output directory if it doesn't exist
    os.makedirs('./predicted', exist_ok=True)
    
    # Generate the output filename based on the input image name
    base_filename = os.path.basename(image_path).rsplit('.', 1)[0]
    output_filename = f'./predicted/{base_filename}_binary_full.png'
    
    # Save the binary image as output
    binary_image.save(output_filename)
    print(f'Prediction saved as {output_filename}')


def load_latest_checkpoint(model, device, checkpoint_dir='./checkpoints'):
    # Find the latest checkpoint file in the checkpoint directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoint_files:
        raise FileNotFoundError(f'No checkpoint files found in {checkpoint_dir}')
    
    # Sort checkpoint files by modification time (latest first)
    latest_checkpoint = max(checkpoint_files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    print(f'Loading latest checkpoint: {checkpoint_path}')
    
    # Load the checkpoint with weights_only=True
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def load_device_and_model():
    global model, device

    # Set device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model
    model = smp.Unet(
        encoder_name="resnet34",        # ResNet34 as the encoder
        encoder_weights="imagenet",     # pretrained weights on ImageNet
        in_channels=3,                  # input channels 3 for RGB images
        classes=1,                      # output channels 1 for binary classification
    ).to(device)

    # Load the latest checkpoint automatically
    model = load_latest_checkpoint(model, device)


# Script execution starts here
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        sys.exit(1)
    
    # Run the prediction
    predict_full_image(image_path)
