import argparse
from models.model_loader import load_model_architecture, load_saved_model
from attacks.finetune import finetune_model
from attacks.pruning import prune_model
from attacks.pruning_finetune import prune_and_finetune
from attacks.model_extraction import model_extraction
# from attacks.adversarial_training import adversarial_training
import torch
import os
from utils.config import config


def main():
    parser = argparse.ArgumentParser(description="Apply attacks to a protected model.")
    parser.add_argument('--model', type=str, required=True, choices=config.get('models', 'architectures'), help='Model architecture')
    parser.add_argument('--pretrained', type=bool, required=True, choices=[True, False], help='Downloaded model or locally trained model')
    parser.add_argument('--seed', type=str, required=True, choices=['1', '...', config.get('models', 'independent_model_count')], help='Model index(seed)')
    parser.add_argument('--attack', type=str, required=True, choices=['finetune', 'pruning', 'pruning_finetune', 'model_extraction', 'adversarial_training'], help='Attack type')
    args = parser.parse_args()
    # model_conf = config.get('models')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name, pretrained, i, save_dir = args.model, args.pretrained, args.seed, config.get('training', 'save_dir')
    model = load_model_architecture(model_name=name, pretrained=pretrained)
    model_path = os.path.join(save_dir, f"{name}_{i}.pt")
    model = load_saved_model(model, model_path)
    target_class = config.get('fingerprint', 'target_class')
    model.to(device)

    if args.attack == 'finetune':
        attack_params = config.get('attacks', 'finetune')
        attacked_model = finetune_model(model, name, args.seed)
    elif args.attack == 'pruning':
        attack_params = config.get('attacks', 'pruning')
        attacked_model = prune_model(model)
    elif args.attack == 'pruning_finetune':
        attack_params = config.get('attacks', 'pruning_finetune')
        attacked_model = prune_and_finetune(model)
    elif args.attack == 'model_extraction':
        attack_params = config.get('attacks', 'model_extraction')
        attacked_model = model_extraction(model)
    elif args.attack == 'adversarial_training':
        attack_params = config.get('attacks', 'adversarial_training')
        attacked_model = adversarial_training(model)
    else:
        raise ValueError("Unsupported attack type.")

if __name__ == "__main__":
    main()
