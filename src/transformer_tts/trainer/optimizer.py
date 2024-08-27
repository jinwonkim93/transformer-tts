import torch

def create_optimizer(model, config):
    config = config.experiment
    optimizer_type = config.optimizer.lower()
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay,
            betas=config.betas
        )
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay, betas=config.betas
        )
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay, momentum=0.9
        )
    else:
        raise ValueError(f'{optimizer_type} invalid..')
    return optimizer