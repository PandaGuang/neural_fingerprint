import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from utils.config import config
import copy
import logging
import random
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """
    Sets the seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For CUDA algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model_architecture(model_name, num_classes=10, pretrained=None):
    """
    Loads the architecture of the specified model without pretrained weights.

    Args:
        model_name (str): Name of the model architecture (e.g., 'vgg16', 'resnet50').
        num_classes (int): Number of output classes for the classifier.
        pretrained (bool): Whether to load pretrained weights. Default is False.

    Returns:
        torch.nn.Module: The model architecture with the modified classifier.
    """
    MODEL_MAPPING = {
        'vgg16': models.vgg16,
        'resnet18': models.resnet18,
        'resnet50': models.resnet50,
        'densenet121': models.densenet121,
        # Add more mappings as needed
    }

    if model_name not in MODEL_MAPPING:
        raise ValueError(f"Model '{model_name}' is not supported.")

    if not pretrained:
        pretrained = None
    model = MODEL_MAPPING[model_name](weights=None)

    # Modify the classifier based on model type
    if 'vgg' in model_name or 'densenet' in model_name:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif 'resnet' in model_name:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    return model


def initialize_weights(model):
    """
    Initializes the weights of the model using Xavier initialization.

    Args:
        model (torch.nn.Module): The model to initialize.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def get_data_loaders():
    """
    Creates training and validation data loaders based on the configuration.

    Returns:
        tuple: (train_loader, val_loader)
    """
    data_conf = config.get('data', default={})
    data_root = data_conf.get('data_root', './data')
    batch_size = data_conf.get('batch_size', 128)
    num_workers = data_conf.get('num_workers', 4)
    num_classes = data_conf.get('num_classes', 10)
    validation_split = config.get('training', default={}).get('validation_split', 0.1)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(data_conf['transforms'].get('resize', 32)),
        transforms.CenterCrop(data_conf['transforms'].get('center_crop', 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_conf['transforms']['normalize']['mean'],
                             std=data_conf['transforms']['normalize']['std'])
    ])

    # Load dataset
    dataset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)

    # Split into training and validation
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, device, training_conf):
    """
    Trains the model and validates it after each epoch.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Device to train on.
        training_conf (dict): Training configuration parameters.

    Returns:
        torch.nn.Module: The trained model.
    """
    # print(device)
    epochs = training_conf.get('epochs', 20)
    learning_rate = training_conf.get('learning_rate', 0.001)
    optimizer_name = training_conf.get('optimizer', 'adam').lower()
    criterion_name = training_conf.get('criterion', 'cross_entropy').lower()
    early_stopping = training_conf.get('early_stopping', {}).get('enabled', False)
    patience = training_conf.get('early_stopping', {}).get('patience', 5)

    # Define loss function
    if criterion_name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif criterion_name == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported criterion '{criterion_name}'.")

    # Define optimizer
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'.")

    best_val_accuracy = 0.0
    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / total
        train_accuracy = 100. * correct / total

        # Validation
        val_loss, val_accuracy = validate_model(model, val_loader, device, criterion)

        logger.info(f"Epoch [{epoch}/{epochs}] "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

        # Check for improvement
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            # Optionally, save the best model state here
        else:
            epochs_no_improve += 1

        # Early Stopping
        if early_stopping and epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered after {epoch} epochs.")
            break

    logger.info(f"Training completed. Best Validation Accuracy: {best_val_accuracy:.2f}%")
    return model


def validate_model(model, val_loader, device, criterion):
    """
    Validates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Device to perform validation on.
        criterion (torch.nn.Module): Loss function.

    Returns:
        tuple: (validation_loss, validation_accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / total
    val_accuracy = 100. * correct / total
    return val_loss, val_accuracy


def save_model(model, model_name, index, save_dir):
    """
    Saves the model's state dictionary to the specified directory.

    Args:
        model (torch.nn.Module): The trained model.
        model_name (str): Name of the model architecture.
        index (int): Index of the model (useful if multiple models are saved).
        save_dir (str): Directory to save the model.
    """
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{model_name}_{index}.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")


def load_saved_model(model, model_path):
    """
    Loads the model's state dictionary from the specified directory.

    Args:
        model (torch.nn.Module): The model architecture to load weights into.
        model_name (str): Name of the model architecture.
        index (int): Index of the model (useful if multiple models are saved).
        save_dir (str): Directory where the model is saved.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model found at {model_path}")

    model.load_state_dict(torch.load(model_path))
    logger.info(f"Loaded model from {model_path}")
    return model


def get_independently_trained_models():
    """
    Retrieves a list of independently initialized and trained models based on the configuration.
    If a saved model exists, it loads the model instead of training a new one.

    Returns:
        list: A list of trained torch.nn.Module models.
    """

    models_conf = config.get('models', default={})
    model_names = models_conf.get('architectures', [])
    count = models_conf.get('independent_model_count', 1)
    pretrained = models_conf.get('pretrained', None)  # Should be False for training from scratch
    num_classes = config.get('data', default={}).get('num_classes', 10)

    training_conf = config.get('training', default={})
    save_dir = training_conf.get('save_dir', 'models/trained')

    # # Setup device
    # print("CUDA Available:", torch.cuda.is_available())
    # print("CUDA Version:", torch.version.cuda)
    # print("Number of GPUs:", torch.cuda.device_count())
    # if torch.cuda.is_available():
    #     print("Current GPU:", torch.cuda.current_device())
    #     print("GPU Name:", torch.cuda.get_device_name(0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    logger.info(f"Using device: {device}")

    # Get data loaders
    train_loader, val_loader = get_data_loaders()

    independent_models = []
    for name in model_names:
        for i in range(1, count + 1):
            # Set the random seed using the model index
            set_seed(i)
            logger.info(f"Setting random seed to {i} for model '{name}' instance {i}.")

            model_filename = f"{name}_{i}.pt"
            model_path = os.path.join(save_dir, model_filename)
            if os.path.exists(model_path):
                logger.info(f"Found saved model '{model_filename}'. Loading it.")
                model = load_model_architecture(model_name=name, num_classes=num_classes, pretrained=pretrained)
                model_path = os.path.join(save_dir, f"{name}_{i}.pt")
                model = load_saved_model(model, model_path)
                model.to(device)
            else:
                logger.info(f"No saved model found for '{name}' instance {i}. Initializing and training a new model.")
                model = load_model_architecture(model_name=name, num_classes=num_classes, pretrained=pretrained)
                initialize_weights(model)
                model = model.to(device)

                logger.info(f"Starting training for model '{name}' (Instance {i}).")
                trained_model = train_model(model, train_loader, val_loader, device, training_conf)
                independent_models.append(trained_model)

                # Save the trained model
                save_model(trained_model, name, i, save_dir)
                continue  # Continue to next model instance

            # If model was loaded, add to the list
            independent_models.append(model)

    return independent_models


if __name__ == "__main__":
    get_independently_trained_models()
