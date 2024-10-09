import argparse
from utils.config import config
from models.model_loader import load_saved_model, load_model_architecture
import torch
import logging



def load_fingerprint(fingerprint_path):
    # Load the fingerprint data, which is a list of fingerprint_entry dictionaries
    return torch.load(fingerprint_path)


def verify_model(model, fingerprint_data, device):
    """
    Verifies the model against the provided fingerprint data.

    Args:
        model (torch.nn.Module): The model to verify.
        fingerprint_data (list): Loaded fingerprint data, a list of fingerprint_entry dictionaries.
        device (torch.device): The device to run the model on.

    Returns:
        float: The percentage of fingerprints correctly matched.
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for fingerprint_entry in fingerprint_data:
            input_tensor = fingerprint_entry['data'].unsqueeze(0).to(device)  # Add batch dimension and move to device
            target_label = fingerprint_entry['label']

            output = model(input_tensor)
            _, pred = torch.max(output, 1)

            if pred.item() == target_label:
                correct += 1
            total += 1

    percentage = (correct / total) * 100
    return percentage


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Verify if a model is the protected model or its variants.")
    parser.add_argument('--model_name', type=str, required=True, help='Model type.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file to verify.')
    parser.add_argument('--fingerprint_path', type=str, required=True, help='Path to the fingerprint file to verify.')
    args = parser.parse_args()

    fingerprint_conf = config.get('fingerprint')
    verification_conf = config.get('verification')
    paths_conf = config.get('paths')

    # Load the fingerprint data
    fingerprint_data = load_fingerprint(args.fingerprint_path)

    # Load the model architecture and weights
    try:
        model_arch = load_model_architecture(model_name=args.model_name)
        model = load_saved_model(model_arch, args.model_path)
        model.eval()
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        exit(1)

    # Move model to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Perform verification
    percentage = verify_model(model, fingerprint_data, device)
    logging.info(f"Fingerprint verification score: {percentage:.2f}%")

    # Determine verification outcome based on the threshold
    threshold = verification_conf.get('threshold', 80.0)  # Default to 80.0 if not specified
    if percentage >= threshold:
        logging.info("Model is likely the protected model or its attacked variants.")
    else:
        logging.info("Model is likely an independently trained model.")


# def main():
#     parser = argparse.ArgumentParser(description="Verify if a model is the protected model or its variants.")
#     parser.add_argument('--model_name', type=str, required=True, help='Model type.')
#     parser.add_argument('--model_path', type=str, required=True, help='Path to the model file to verify.')
#     parser.add_argument('--fingerprint_path', type=str, required=True, help='Path to the fingerprint file to verify.')
#     args = parser.parse_args()
#
#     fingerprint_conf = config.get('fingerprint')
#     verification_conf = config.get('verification')
#     paths_conf = config.get('paths')
#
#     # Load the fingerprint data
#     fingerprint_data = load_fingerprint(args.fingerprint_path)
#
#     # Load the model architecture and weights
#     model_arch = load_model_architecture(model_name=args.model_name)
#     model = load_saved_model(model_arch, args.model_path)
#     model.eval()
#
#     # Move model to the appropriate device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#
#     # Perform verification
#     percentage = verify_model(model, fingerprint_data, device)
#     print(f"Fingerprint verification score: {percentage:.2f}%")
#
#     # Determine verification outcome based on the threshold
#     threshold = verification_conf.get('threshold', 80.0)  # Default to 80.0 if not specified
#     if percentage >= threshold:
#         print("Model is likely the protected model or its attacked variants.")
#     else:
#         print("Model is likely an independently trained model.")


if __name__ == "__main__":
    main()
