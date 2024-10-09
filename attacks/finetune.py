import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_loader import get_cifar10_data #get_imagenet_data
from utils.config import config
import os

def save_model(model, attack_type, attacked_models_dir, name, seed, epoch, accuracy):
    save_path = os.path.join(attacked_models_dir, f"{name}_{seed}_finetune_epoch{epoch}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved attacked model: {save_path}")
    log_file = os.path.join(attacked_models_dir, f"{name}_{seed}_{attack_type}_{epoch}.txt")
    with open(log_file, 'a') as f:
        f.write(f"Epoch {epoch}: Test Accuracy = {accuracy:.4f}\n")
    print(f"Epoch {epoch}: Test Accuracy = {accuracy:.4f} (Logged to {log_file})")

def finetune_model(model, name, seed):
    attack_conf = config.get('attacks', 'finetune')
    epochs = attack_conf.get('epochs')
    learning_rate = attack_conf.get('learning_rate')
    interval = attack_conf.get('interval')
    # performance_constraint = attack_conf.get('performance_constraint')

    attacked_models_dir = config.get('paths', 'attacked_models_dir')
    os.makedirs(attacked_models_dir, exist_ok=True)

    train_loader, test_loader = get_cifar10_data()
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
        if (epoch+1) % interval == 0:
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            accuracy = correct / total

            save_model(model, "finetune", attacked_models_dir, name, seed, epoch+1, accuracy)
            # print(f"Epoch {epoch + 1}: Test Accuracy = {accuracy:.4f}")

        # if accuracy < performance_constraint:
        #     print("Performance constraint breached. Stopping fine-tuning.")
        #     break
    return model
