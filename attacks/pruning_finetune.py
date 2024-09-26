from attacks.pruning import prune_model
from attacks.finetune import finetune_model
from utils.config import config


def prune_and_finetune(model):
    attack_conf = config.get('attacks', 'pruning_finetune')
    pruning_amount = attack_conf.get('pruning_amount')
    finetune_epochs = attack_conf.get('finetune_epochs')
    learning_rate = attack_conf.get('learning_rate')
    performance_constraint = attack_conf.get('performance_constraint')

    # Apply pruning
    pruned_model = prune_model_custom(model, pruning_amount, performance_constraint)
    if pruned_model is None:
        return None

    # Apply fine-tuning
    finetuned_model = finetune_model_custom(pruned_model, finetune_epochs, learning_rate, performance_constraint)
    return finetuned_model


def prune_model_custom(model, pruning_amount, performance_constraint):
    from attacks.pruning import prune_model
    # Temporarily override the config for pruning parameters
    original_pruning_amount = config.get('attacks', 'pruning', 'pruning_amount')
    original_performance_constraint = config.get('attacks', 'pruning', 'performance_constraint')

    # Update config with new parameters
    config.config['attacks']['pruning']['pruning_amount'] = pruning_amount
    config.config['attacks']['pruning']['performance_constraint'] = performance_constraint

    pruned_model = prune_model(model)

    # Restore original config
    config.config['attacks']['pruning']['pruning_amount'] = original_pruning_amount
    config.config['attacks']['pruning']['performance_constraint'] = original_performance_constraint

    return pruned_model


def finetune_model_custom(model, epochs, learning_rate, performance_constraint):
    from attacks.finetune import finetune_model
    # Temporarily override the config for finetuning parameters
    original_epochs = config.get('attacks', 'finetune', 'epochs')
    original_learning_rate = config.get('attacks', 'finetune', 'learning_rate')
    original_performance_constraint = config.get('attacks', 'finetune', 'performance_constraint')

    # Update config with new parameters
    config.config['attacks']['finetune']['epochs'] = epochs
    config.config['attacks']['finetune']['learning_rate'] = learning_rate
    config.config['attacks']['finetune']['performance_constraint'] = performance_constraint

    finetuned_model = finetune_model(model)

    # Restore original config
    config.config['attacks']['finetune']['epochs'] = original_epochs
    config.config['attacks']['finetune']['learning_rate'] = original_learning_rate
    config.config['attacks']['finetune']['performance_constraint'] = original_performance_constraint

    return finetuned_model
