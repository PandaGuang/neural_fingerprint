import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_loader import get_cifar10_data # get_imagenet_data
from utils.config import config
import copy
import foolbox as fb
from foolbox import PyTorchModel
from torch.optim import lr_scheduler


def adversarial_training(model):
    attack_conf = config.get('attacks', 'adversarial_training')
    epochs = attack_conf.get('epochs')
    learning_rate = attack_conf.get('learning_rate')
    performance_constraint = attack_conf.get('performance_constraint')
    epsilon = attack_conf.get('epsilon')

    train_loader, test_loader = get_cifar10_data()
    model = copy.deepcopy(model)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Initialize Foolbox
    fmodel = PyTorchModel(model, bounds=(0, 1))
    attack = fb.attacks.LinfPGD()

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
            optimizer.zero_grad()

            # Generate adversarial examples
            adversarial = attack(fmodel, inputs, labels, epsilons=epsilon)
            adversarial = adversarial.tensor

            # Forward pass with adversarial examples
            outputs = model(adversarial)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Evaluate performance
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
        print(f"Epoch {epoch + 1}: Adversarially Trained Model Test Accuracy = {accuracy:.4f}")

        if accuracy < performance_constraint:
            print("Performance constraint breached. Stopping adversarial training.")
            break
        # model.train()
    return model
