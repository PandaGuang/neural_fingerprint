import torch
from torchvision import transforms
from PIL import Image
import os
from utils.config import config


def generate_fingerprint():
    fingerprint_conf = config.get('fingerprint')
    num_samples = fingerprint_conf.get('num_samples')
    save_dir = fingerprint_conf.get('save_dir')
    generation_method = fingerprint_conf.get('generation_method')

    os.makedirs(save_dir, exist_ok=True)
    fingerprint_data = []

    if generation_method == 'random_noise':
        for i in range(num_samples):
            noise = torch.randn(3, 32, 32)  # Random noise: [3, 224, 224] for imagenet
            fingerprint_data.append(noise)
            # Optionally save to disk
            # torchvision.utils.save_image(noise, os.path.join(save_dir, f'fingerprint_{i}.png'))
    elif generation_method == 'adversarial_example':
        # Placeholder for future implementation
        raise NotImplementedError("Adversarial example generation not implemented yet.")
    else:
        raise NotImplementedError(f"Fingerprint generation method '{generation_method}' is not implemented.")

    torch.save(fingerprint_data, os.path.join(save_dir, 'fingerprint.pt'))
    print(f"Generated {num_samples} fingerprint samples using method '{generation_method}'.")


if __name__ == "__main__":
    generate_fingerprint()
