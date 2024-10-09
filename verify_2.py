import argparse
import os
from utils.config import config
from models.model_loader import load_saved_model, load_model_architecture
import torch

def load_fingerprint_data(fingerprint_path):
    """
    Load the fingerprint data from the specified path.

    Args:
        fingerprint_path (str): Path to the fingerprint file.

    Returns:
        list: A list of fingerprint_entry dictionaries.
    """
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

    percentage = (correct / total) * 100 if total > 0 else 0.0
    return percentage

def main():
    parser = argparse.ArgumentParser(description="Verify if models in a folder are the protected model or its variants.")
    parser.add_argument('--model_name', type=str, required=True, help='Model name: resnet18.')
    parser.add_argument('--model_type', type=str, required=True, help='Trained models or attacked models.')
    parser.add_argument('--fingerprint_type', type=str, required=True, help='Type of the fingerprint for naming the result file.')
    parser.add_argument('--seed', type=str, required=True, help='Seed.')
    args = parser.parse_args()

    # fingerprint_conf = config.get('fingerprint')
    verification_conf = config.get('verification')
    paths_conf = config.get('paths')

    # Load the fingerprint data
    fingerprint_path = os.path.join(paths_conf.get("fingerprint_dir"), f"{args.model_name}_{args.seed}_{args.fingerprint_type}_fingerprints.pt")
    fingerprint_data = load_fingerprint_data(fingerprint_path)

    # Load the model architecture
    model_arch = load_model_architecture(model_name=args.model_name)

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare to iterate over all model files in the specified folder
    models_folder = paths_conf.get(f"{args.model_type}_models_dir")
    if not os.path.isdir(models_folder):
        print(f"Error: The specified models_folder '{models_folder}' is not a directory.")
        return

    # Define acceptable model file extensions
    model_extensions = ['.pt', '.pth', '.model', '.bin']  # Adjust as needed

    # Collect all model files
    model_files = [f for f in os.listdir(models_folder) if os.path.isfile(os.path.join(models_folder, f)) and os.path.splitext(f)[1] in model_extensions]

    if not model_files:
        print(f"No model files found in the directory '{models_folder}'.")
        return

    # Prepare the output file path
    output_file_name = f"verification_results_{args.fingerprint_type}.txt"
    output_file_path = os.path.join(models_folder, output_file_name)

    with open(output_file_path, 'w') as output_file:
        output_file.write(f"Fingerprint Verification Results (Type: {args.fingerprint_type})\n")
        output_file.write(f"{'Model Filename':<50} {'Verification Score (%)':>20}\n")
        output_file.write("-" * 72 + "\n")

        for model_file in model_files:
            model_path = os.path.join(models_folder, model_file)
            try:
                # Load the saved model
                model = load_saved_model(model_arch, model_path)
                model.eval()
                model.to(device)

                # Perform verification
                percentage = verify_model(model, fingerprint_data, device)
                output_file.write(f"{model_file:<50} {percentage:>20.2f}\n")

                print(f"Verified '{model_file}': {percentage:.2f}%")

            except Exception as e:
                # Handle any errors during model loading or verification
                error_message = f"Error verifying '{model_file}': {str(e)}"
                output_file.write(f"{model_file:<50} {'Error':>20}\n")
                print(error_message)

        output_file.write("\nVerification completed.\n")

    # Determine verification outcome based on the threshold for each model
    threshold = verification_conf.get('threshold', 80.0)  # Default to 80.0 if not specified
    print(f"\nVerification results saved to '{output_file_path}'.")
    print(f"Threshold for verification: {threshold}%")

if __name__ == "__main__":
    main()
