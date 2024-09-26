import argparse
from utils.config import config
from fingerprints.generate_fingerprint import generate_fingerprint
from attack_models import main as attack_main
from verify import main as verify_main
import sys

def main():
    parser = argparse.ArgumentParser(description="Neural Network Fingerprinting Project")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Subparser for generating fingerprints
    parser_fingerprint = subparsers.add_parser('generate_fingerprint', help='Generate fingerprint data')

    # Subparser for applying attacks
    parser_attack = subparsers.add_parser('attack', help='Apply attacks to models')
    parser_attack.add_argument('--model', type=str, required=True, choices=config.get('models', 'architectures'), help='Model architecture')
    parser_attack.add_argument('--attack', type=str, required=True, choices=['finetune', 'pruning', 'pruning_finetune', 'model_extraction', 'adversarial_training'], help='Attack type')

    # Subparser for verification
    parser_verify = subparsers.add_parser('verify', help='Verify a model against fingerprints')
    parser_verify.add_argument('--model_path', type=str, required=True, help='Path to the model file to verify.')

    args = parser.parse_args()

    if args.command == 'generate_fingerprint':
        generate_fingerprint()
    elif args.command == 'attack':
        # Redirect arguments to attack_models.py
        sys.argv = ['attack_models.py', '--model', args.model, '--attack', args.attack]
        attack_main()
    elif args.command == 'verify':
        # Redirect arguments to verify.py
        sys.argv = ['verify.py', '--model_path', args.model_path]
        verify_main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
