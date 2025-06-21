import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


class HuggingFaceAutoEncoder(nn.Module):
    def __init__(self, model_name="stabilityai/sd-vae-ft-mse", device=None):
        super().__init__()
        self.ae = AutoencoderKL.from_pretrained(model_name)
        if device:
            self.ae = self.ae.to(device)
        self.ae.eval()
        
        for param in self.ae.parameters():
            param.requires_grad = False
            
    def encode(self, x):
        posterior = self.ae.encode(x).latent_dist
        latents = posterior.sample() * self.ae.config.scaling_factor
        return latents
            
    def decode(self, latents):
        # Scale latents
        latents = latents / self.ae.config.scaling_factor
        decoded = self.ae.decode(latents).sample
        return decoded
            
    def forward(self, x, encode=True):
        if encode:
            return self.encode(x)
        else:
            return self.decode(x)


def test_autoencoder():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    ae = HuggingFaceAutoEncoder(device=device)
    
    # Load and preprocess image
    img = Image.open("/Users/souymodip/GIT/VectorizationTest/TEST_SUITS/pix_art/2.png").convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    x = transform(img).unsqueeze(0).to(device)
    
    # Encode and decode
    encoded = ae.encode(x)
    decoded = ae.decode(encoded)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(x.squeeze(0).cpu().permute(1, 2, 0))
    ax1.set_title('Input')
    
    # Denormalize and convert to PIL for display
    decoded_img = (decoded.squeeze(0).cpu().permute(1, 2, 0))
    decoded_img = decoded_img.clamp(0, 1)

    ax2.imshow(decoded_img)
    ax2.set_title('Reconstructed')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_autoencoder() 