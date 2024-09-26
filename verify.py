import argparse
from utils.config import config
from utils.data_loader import get_imagenet_data
from fingerprints.generate_fingerprint import load_fingerprint
from models.model_loader import load_pretrained_model
import torch

def load_fingerprint(fingerprint_path):
    return torch.load(fingerprint_path)

def verify_model(model, fingerprint_data, target_class, device):
    model.to(device)
    model.eval()
    correct = 0
    total = len(fingerprint_data)
    with torch.no_grad():
        for input_tensor in fingerprint_data:
            input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to device
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            if pred.item() == target_class:
                correct += 1
    percentage = (correct / total) * 100
    return percentage

def main():
    parser = argparse.ArgumentParser(description="Verify if a model is the protected model or its variants.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file to verify.')
    args = parser.parse_args()

    fingerprint_conf = config.get('fingerprint')
    verification_conf = config.get('verification')
    paths_conf = config.get('paths')

    fingerprint_data = load_fingerprint(paths_conf.get('fingerprint_file'))
    target_class = fingerprint_conf.get('target_class')

    # Load the model
    model_architectures = config.get('models', 'architectures', default=['resnet50'])
    if len(model_architectures) == 1:
        model_arch = model_architectures[0]
    else:
        # If multiple architectures, specify or infer from model_path
        raise NotImplementedError("Multiple architectures are not supported in verification yet.")

    model = load_pretrained_model(model_arch)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Move model to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    percentage = verify_model(model, fingerprint_data, target_class, device)
    print(f"Fingerprint verification score: {percentage:.2f}%")

    threshold = verification_conf.get('threshold', 80.0)  # Default to 80.0 if not specified
    if percentage >= threshold:
        print("Model is likely the protected model or its attacked variants.")
    else:
        print("Model is likely an independently trained model.")

if __name__ == "__main__":
    main()
