import argparse
from models.model_loader import load_pretrained_model
from attacks.finetune import finetune_model
from attacks.pruning import prune_model
from attacks.pruning_finetune import prune_and_finetune
from attacks.model_extraction import model_extraction
from attacks.adversarial_training import adversarial_training
import torch
import os
from utils.config import config

def save_model(model, attack_type, model_name):
    attacked_models_dir = config.get('paths', 'attacked_models_dir')
    os.makedirs(attacked_models_dir, exist_ok=True)
    save_path = os.path.join(attacked_models_dir, f"{model_name}_{attack_type}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved attacked model: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Apply attacks to a protected model.")
    parser.add_argument('--model', type=str, required=True, choices=config.get('models', 'architectures'), help='Model architecture')
    parser.add_argument('--attack', type=str, required=True, choices=['finetune', 'pruning', 'pruning_finetune', 'model_extraction', 'adversarial_training'], help='Attack type')
    args = parser.parse_args()

    model_name = args.model
    attack_type = args.attack

    # Load the protected model
    model = load_pretrained_model(model_name)

    # Apply the specified attack
    if attack_type == 'finetune':
        attacked_model = finetune_model(model)
    elif attack_type == 'pruning':
        attacked_model = prune_model(model)
    elif attack_type == 'pruning_finetune':
        attacked_model = prune_and_finetune(model)
    elif attack_type == 'model_extraction':
        attacked_model = model_extraction(model)
    elif attack_type == 'adversarial_training':
        attacked_model = adversarial_training(model)
    else:
        raise ValueError("Unsupported attack type.")

    # Save the attacked model if it meets performance constraints
    if attacked_model is not None:
        save_model(attacked_model, attack_type, model_name)
    else:
        print("Attack resulted in model exceeding performance constraints. Model not saved.")

if __name__ == "__main__":
    main()
