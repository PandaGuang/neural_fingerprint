import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_loader import get_imagenet_data
from utils.config import config


def finetune_model(model):
    attack_conf = config.get('attacks', 'finetune')
    epochs = attack_conf.get('epochs')
    learning_rate = attack_conf.get('learning_rate')
    performance_constraint = attack_conf.get('performance_constraint')

    train_loader, test_loader = get_imagenet_data()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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
        print(f"Epoch {epoch + 1}: Test Accuracy = {accuracy:.4f}")

        if accuracy < performance_constraint:
            print("Performance constraint breached. Stopping fine-tuning.")
            break
    return model
