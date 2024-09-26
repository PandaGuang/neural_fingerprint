# Neural Network Fingerprinting
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

Neural Network Fingerprinting is a technique to protect the intellectual property (IP) of neural network models by distinguishing them from independently trained models using specially designed input data (fingerprints).

## Features

- **Fingerprint Generation**: Create fingerprint input data that triggers specific behaviors in protected models.
- **Model Attacks**: Apply various attacks (fine-tuning, pruning, model extraction, adversarial training) to generate model variants.
- **Fingerprint Verification**: Verify if a given model is the protected model or its attacked variants.
- **Metrics Evaluation**: Assess the effectiveness of fingerprinting using relevant metrics.

## Datasets
- **CIFAR-10:** A dataset consisting of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Configuration

All configurable parameters are centralized in the `config.json` file located at the root of the project. This includes settings for data loading, model architectures, fingerprint generation, attack parameters, verification thresholds, and file paths.

### Editing Configuration

To modify parameters:

1. Open the `config.json` file.
2. Navigate to the relevant section (e.g., `attacks`, `fingerprint`, `verification`).
3. Adjust the values as needed.

**Example:** To change the number of fingerprint samples, update the `num_samples` field under the `fingerprint` section.

```json
"fingerprint": {
    "num_samples": 150,
    "save_dir": "fingerprints",
    "target_class": 123,
    "generation_method": "random_noise"
}
```
### Configuration Categories

**Data**: Parameters related to dataset loading and preprocessing.
**Models**: Specifications for model architectures and independent model generation.
**Fingerprint**: Settings for fingerprint generation, including the number of samples, save directory, target class, and generation method.
**Attacks**: Detailed parameters for each attack type.
**Verification**: Threshold settings for determining model authenticity.
**Paths**: File and directory paths used across the project.

## Installation
Ensure you have Conda installed. Then, create and activate the environment:
```bash
conda env create -f environment.yml
conda activate neural_fingerprint_env
```

## Usage
1. Generate Fingerprints
```bash
python generate_fingerprint.py
```
2. Apply Attacks to the Protected Model
```bash
python attack_models.py --model resnet50 --attack finetune
python attack_models.py --model resnet50 --attack pruning
python attack_models.py --model resnet50 --attack pruning_finetune
python attack_models.py --model resnet50 --attack model_extraction
python attack_models.py --model resnet50 --attack adversarial_training
```
**Parameters:**
- ```--model```: specify the model architecture (e.g., ```resnet50```, ```vgg16```, ```densenet121```)
- ```--attack```: choose the attack type (```finetune```, ```pruning```, ```pruning_finetune```, ```model_extraction```, ```adversarial_training```)
3. Verify a Model 
```bash
python verify.py --model_path models/attacked/resnet18_finetune.pth
```
**Process:**
- The script loads fingerprint data and verification threshold from config.json.
- Evaluates the model's responses to fingerprint inputs.
- Determines authenticity based on the configured threshold.

4. Run Tests
Ensure all components are functioning correctly.
```bash
python -m unittest discover tests
```

## Future Enhancements
- Implement additional fingerprint schemes.
- Extend support to other tasks beyond image classification.
- Incorporate more sophisticated fingerprint generation techniques.

## License
This project is licensed under the [MIT License](LICENSE).