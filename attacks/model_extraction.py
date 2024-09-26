import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_loader import get_imagenet_data
from utils.config import config
import copy


def model_extraction(original_model):
    attack_conf = config.get('attacks', 'model_extraction')
    epochs = attack_conf.get('epochs')
    learning_rate = attack_conf.get('learning_rate')
    performance_constraint = attack_conf.get('performance_constraint')

    # Define a student model (same architecture)
    student_model = copy.deepcopy(original_model)
    student_model.train()

    train_loader, test_loader = get_imagenet_data()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(next(student_model.parameters()).device), labels.to(
                next(student_model.parameters()).device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_outputs = original_model(inputs)
            # Optionally, use knowledge distillation loss here
            loss = criterion(student_model(inputs), labels)  # Simplified loss
            loss.backward()
            optimizer.step()

        # Evaluate performance
        student_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(next(student_model.parameters()).device), labels.to(
                    next(student_model.parameters()).device)
                outputs = student_model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        print(f"Epoch {epoch + 1}: Student Model Test Accuracy = {accuracy:.4f}")

        if accuracy < performance_constraint:
            print("Performance constraint breached. Stopping model extraction.")
            break
        student_model.train()
    return student_model
