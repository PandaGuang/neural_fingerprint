import torch
import argparse
from torchvision import transforms, datasets
import os
from utils.config import config
from models.model_loader import load_model_architecture, load_saved_model  # Assuming this function is accessible
import logging
from torch.utils.data import DataLoader, Subset
import random
from torch import nn, optim

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_fingerprint():
    """
    Generates fingerprints by performing adversarial attacks or optimized random sampling
    on sampled training data using the saved models. The fingerprints consist of perturbed
    data points and their new (incorrect) labels or target labels.
    """
    # Load fingerprint configuration
    fingerprint_conf = config.get('fingerprint', default={})
    num_samples = fingerprint_conf.get('num_samples', 100)
    num_fps = fingerprint_conf.get('num_fps', 20)
    fingerprints_dir = config.get('paths', 'fingerprint_dir')
    # generation_method = fingerprint_conf.get('generation_method', 'random_noise')
    target_class = fingerprint_conf.get('target_class', None)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Apply optimized random fingerprint generation.")
    parser.add_argument('--model', type=str, required=True, choices=config.get('models', 'architectures'),
                        help='Model architecture')
    parser.add_argument('--pretrained', type=bool, required=True, choices=[True, False],
                        help='Downloaded model or locally trained model')
    parser.add_argument('--seed', type=str, required=True,
                        choices=[str(i) for i in range(1, int(config.get('models', 'independent_model_count')) + 1)],
                        help='Model index(seed)')
    parser.add_argument('--generation_method', type=str, required=True, choices={'adversarial_example', 'optimized_random'}, help='The fingerprint types.')
    args = parser.parse_args()
    generation_method = args.generation_method
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name, pretrained, model_idx, save_dir = args.model, args.pretrained, args.seed, config.get('training', 'save_dir')
    model = load_model_architecture(model_name=name, pretrained=pretrained)
    model_path = os.path.join(save_dir, f"{name}_{model_idx}.pt")
    model = load_saved_model(model, model_path)
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Load training data to determine input range
    data_conf = config.get('data', default={})
    data_root = data_conf.get('data_root', './data')
    batch_size = data_conf.get('batch_size', 128)
    num_workers = data_conf.get('num_workers', 1)

    transform = transforms.Compose([
        transforms.Resize(data_conf['transforms'].get('resize', 32)),
        transforms.CenterCrop(data_conf['transforms'].get('center_crop', 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_conf['transforms']['normalize']['mean'],
                             std=data_conf['transforms']['normalize']['std'])
    ])

    full_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)

    # Create save directory if it doesn't exist
    os.makedirs(fingerprints_dir, exist_ok=True)


    if generation_method == 'adversarial_example':
        # Existing adversarial example generation method
        logger.info("Starting adversarial fingerprint generation.")

        # Load adversarial attack parameters
        attack_conf = fingerprint_conf.get('adversarial_attack', {})
        epsilon = attack_conf.get('epsilon', 0.03)
        num_steps = attack_conf.get('num_steps', 10)
        step_size = attack_conf.get('step_size', 0.01)

        # Randomly sample num_samples indices
        all_indices = list(range(len(full_dataset)))
        sampled_indices = random.sample(all_indices, num_samples)
        subset = Subset(full_dataset, sampled_indices)
        data_loader = DataLoader(subset, batch_size=num_samples, shuffle=False, num_workers=num_workers)

        logger.info(f"Sampling {num_samples} data points for fingerprint generation.")

        # Iterate over sampled data
        for idx, (data, label) in enumerate(data_loader):
            # Clone data and set requires_grad
            perturbed_data = data.clone().to(device).requires_grad_(True)
            target_label = label.to(device)
            data = data.to(device)
            # Perform PGD attack
            for step in range(num_steps):
                model.zero_grad()
                output = model(perturbed_data)
                loss = nn.CrossEntropyLoss()(output, target_label)
                loss.backward()

                # Update perturbed data by ascending the gradient
                with torch.no_grad():
                    perturbation = step_size * perturbed_data.grad.sign()
                    perturbed_data += perturbation
                    # Clamp to ensure perturbation is within epsilon
                    perturbed_data = torch.max(torch.min(perturbed_data, data + epsilon), data - epsilon)
                    # Re-normalize if necessary
                    perturbed_data = torch.clamp(perturbed_data, 0, 1)
                    perturbed_data.requires_grad = True

        # Final prediction after attack
        final_output = model(perturbed_data)
        final_pred = final_output.max(1, keepdim=True)[1]
        # Initialize fingerprint entry
        fingerprint_entry = []
        # Check if attack was successful
        for j in range(num_samples):
            if final_pred[j].item() != target_label[j].item():
                logger.info(f"Sample {j + 1}: Model {name}_{model_idx} misclassified after attack.")
                # Store the perturbed data and new label
                fingerprint = {
                    'data': perturbed_data[j,:].detach().cpu(),
                    'label': final_pred[j].item()
                }
                fingerprint_entry.append(fingerprint)
                if len(fingerprint_entry) >= num_fps:
                    break
            else:
                logger.info(f"Sample {j + 1}: Model {name}_{model_idx} failed to misclassify after attack.")

        # Save all fingerprints to a single file
        fingerprint_filepath = os.path.join(fingerprints_dir, f'{name}_{model_idx}_adversarial_example_fingerprints.pt')
        torch.save(fingerprint_entry, fingerprint_filepath)
        logger.info(f"Generated and saved {len(fingerprint_entry)} fingerprints to '{fingerprint_filepath}'.")

    elif generation_method == 'optimized_random':
        # New optimized random fingerprint generation method
        logger.info("Starting optimized random fingerprint generation.")

        # Load optimization parameters
        optimization_conf = fingerprint_conf.get('optimization', {})
        learning_rate = optimization_conf.get('learning_rate', 0.01)
        num_iterations = optimization_conf.get('num_iterations', 100)
        lambda_grad = optimization_conf.get('lambda_grad', 0.001)

        if target_class is None:
            raise ValueError("For 'optimized_random' generation_method, 'target_class' must be specified in the configuration.")

        # Determine per-channel min and max after normalization
        logger.info("Determining input range from the dataset.")
        channel_min = torch.full((3,), float('inf'))
        channel_max = torch.full((3,), float('-inf'))

        data_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        for data, _ in data_loader:
            # data shape: (batch_size, 3, H, W)
            current_min = data.view(data.size(0), data.size(1), -1).min(2)[0].min(0)[0]
            current_max = data.view(data.size(0), data.size(1), -1).max(2)[0].max(0)[0]
            channel_min = torch.min(channel_min, current_min)
            channel_max = torch.max(channel_max, current_max)

        logger.info(f"Per-channel normalized min: {channel_min}")
        logger.info(f"Per-channel normalized max: {channel_max}")

        # Sample uniformly random fingerprints within the determined range
        logger.info(f"Sampling {num_samples} random fingerprints within the determined range.")
        # for i in range(num_samples):
        # Sample uniformly for each channel
        random_input = torch.zeros(num_samples, 3, 32, 32)
        for c in range(3):
            random_input[:, c, :, :] = torch.empty(num_samples, 32, 32).uniform_(channel_min[c].item(), channel_max[c].item())
        random_input = random_input.to(device).requires_grad_(True)

        # Initialize optimizer
        optimizer_fingerprint = optim.Adam([random_input], lr=learning_rate)

        for iteration in range(num_iterations):
            optimizer_fingerprint.zero_grad()
            output = model(random_input)
            prob = nn.Softmax(dim=1)(output)
            target_prob = prob[:, target_class]

            # Calculate gradients
            # Objective 1: Maximize the probability of the target class
            loss_prob = -torch.log(target_prob + 1e-8).mean()  # Mean over batch, Adding epsilon to prevent log(0)

            # Objective 2: Minimize the gradient norm (output smoothness)
            # Compute gradient of target_prob w.r.t input
            grad_outputs = torch.ones_like(target_prob, device=device)
            gradients = torch.autograd.grad(
                outputs=target_prob,
                inputs=random_input,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=True,
                only_inputs=True
            )[0]
            # Reshape gradients to [num_samples, -1] to compute the norm per sample
            gradients_flat = gradients.view(num_samples, -1)
            # Compute the L2 norm for each sample's gradient
            grad_norms = gradients_flat.norm(p=2, dim=1)
            # Compute the mean of the gradient norms
            loss_grad = grad_norms.mean()

            # Combined loss
            loss = loss_prob + lambda_grad * loss_grad

            # Backward pass
            loss.backward()

            # Optimization step
            optimizer_fingerprint.step()

            # Clamp the input to stay within the determined range
            with torch.no_grad():
                for c in range(3):
                    random_input[:, c, :, :].clamp_(channel_min[c].item(), channel_max[c].item())

            if (iteration + 1) % 20 == 0 or iteration == 0:
                logger.info(f"Fingerprints, Iteration {iteration + 1}/{num_iterations}, "
                             f"Loss: {loss.item():.4f}, Target Prob: {target_prob.mean().item():.4f}, "
                             f"Grad Norm: {loss_grad.item():.4f}")

        # After optimization, evaluate the final prediction
        with torch.no_grad():
            final_output = model(random_input)
            final_prob = nn.Softmax(dim=1)(final_output)
            # final_pred = final_prob.max(1, keepdim=True)[1].item()
            final_pred = final_prob.argmax(dim=1).cpu().tolist()
            final_prob_values = final_prob[:, target_class].cpu().tolist()
        # Store all fingerprints with their probabilities
        all_fingerprints = []
        for i in range(num_samples):
            fingerprint_entry = {
                'data': random_input[i].detach().cpu(),
                'label': final_pred[i],  # The predicted label (ideally, should be target_class)
                'prob': final_prob_values[i]  # Probability of the target class
            }
            all_fingerprints.append(fingerprint_entry)
            logger.info(
                f"Generated fingerprint {i + 1}/{num_samples} with label {final_pred[i]} and prob {final_prob_values[i]:.4f}.")

        # Select top num_fps fingerprints based on target class probability
        logger.info(f"Selecting top {num_fps} fingerprints based on target class probability.")
        # Sort the fingerprints in descending order of 'prob'
        all_fingerprints_sorted = sorted(all_fingerprints, key=lambda x: x['prob'], reverse=True)
        top_fingerprints = all_fingerprints_sorted[:num_fps]

        # Prepare the fingerprint_data structure similar to adversarial_example
        fingerprint_entry = []
        for fp in top_fingerprints:
            fingerprint = {
                'data': fp['data'],
                'label': fp['label']
            }
            fingerprint_entry.append(fingerprint)

        # Save all fingerprints to a single file
        fingerprint_filepath = os.path.join(fingerprints_dir, f'{name}_{model_idx}_optimized_random_fingerprints.pt')
        torch.save(fingerprint_entry, fingerprint_filepath)
        logger.info(f"Generated and saved {len(top_fingerprints)} optimized random fingerprints to '{fingerprint_filepath}'.")
    else:
        raise ValueError(f"Unsupported generation_method: {generation_method}")


if __name__ == "__main__":
    generate_fingerprint()
