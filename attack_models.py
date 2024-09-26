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

    model_conf = config.get('models')
    model_name = args.model
    model = load_pretrained_model(model_name)
    target_class = config.get('fingerprint', 'target_class')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if args.attack == 'finetune':
        attack_params = config.get('attacks', 'finetune')
        attacked_model = finetune_model(
            model,
            epochs=attack_params.get('epochs'),
            learning_rate=attack_params.get('learning_rate'),
            performance_constraint=attack_params.get('performance_constraint')
        )
    elif args.attack == 'pruning':
        attack_params = config.get('attacks', 'pruning')
        attacked_model = prune_model(
            model,
            pruning_amount=attack_params.get('pruning_amount'),
            performance_constraint=attack_params.get('performance_constraint')
        )
    elif args.attack == 'pruning_finetune':
        attack_params = config.get('attacks', 'pruning_finetune')
        attacked_model = prune_and_finetune(
            model,
            pruning_amount=attack_params.get('pruning_amount'),
            finetune_epochs=attack_params.get('finetune_epochs'),
            learning_rate=attack_params.get('learning_rate'),
            performance_constraint=attack_params.get('performance_constraint')
        )
    elif args.attack == 'model_extraction':
        attack_params = config.get('attacks', 'model_extraction')
        attacked_model = model_extraction(
            model,
            epochs=attack_params.get('epochs'),
            learning_rate=attack_params.get('learning_rate'),
            performance_constraint=attack_params.get('performance_constraint')
        )
    elif args.attack == 'adversarial_training':
        attack_params = config.get('attacks', 'adversarial_training')
        attacked_model = adversarial_training(
            model,
            epochs=attack_params.get('epochs'),
            learning_rate=attack_params.get('learning_rate'),
            performance_constraint=attack_params.get('performance_constraint'),
            epsilon=attack_params.get('epsilon')
        )
    else:
        raise ValueError("Unsupported attack type.")

    if attacked_model is not None:
        save_model(attacked_model, args.attack, model_name)
    else:
        print("Attack resulted in model exceeding performance constraints. Model not saved.")

if __name__ == "__main__":
    main()
