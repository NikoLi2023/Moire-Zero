import torch.optim as optim


def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(0.9, 0.9), amsgrad=config.optim.amsgrad, eps=config.optim.eps)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay,
                          betas=(0.9, 0.9), amsgrad=config.optim.amsgrad, eps=config.optim.eps)
    elif config.optim.optimizer == 'RMSProp':
        optimizer = optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        optimizer = optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.training.n_epochs, eta_min=1e-6, last_epoch=-1,)
    return optimizer, scheduler
