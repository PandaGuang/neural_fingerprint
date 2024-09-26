import torch
import torch.nn.utils.prune as prune
from utils.data_loader import get_cifar10_data # get_imagenet_data
from utils.config import config
import copy


def prune_model(model):
    attack_conf = config.get('attacks', 'pruning')
    pruning_amount = attack_conf.get('pruning_amount')
    performance_constraint = attack_conf.get('performance_constraint')

    train_loader, test_loader = get_cifar10_data()
    model = copy.deepcopy(model)  # Avoid modifying original model

    # Apply global unstructured pruning
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_amount,
    )

    # Remove pruning re-parametrization
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')

    # Evaluate performance
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('attacks', 'finetune', 'learning_rate'))

    model.train()
    for epoch in range(1):  # Single epoch for pruning
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Pruned Model Test Accuracy = {accuracy:.4f}")

    if accuracy < performance_constraint:
        print("Performance constraint breached after pruning.")
        return None
    return model
