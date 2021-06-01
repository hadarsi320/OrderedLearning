from torch import optim

import training
import utils
from data import imagenette
from models.classifiers import Classifier


# TODO update this module to work with configuration yaml files
# TODO stop using general_train module
def train_classifier():
    # data options
    normalize_data = True
    loss_criterion = 'CrossEntropyLoss'
    dataset = 'imagenette'
    image_mode = 'Y'
    channels = 1 if image_mode == 'Y' else 3

    # training options
    optim_alg = 'SGD'
    epochs = 100
    learning_rate = 1e-1
    patience = 5
    batch_size = 32
    plateau_limit = 10

    # model options
    activation = 'ReLU'
    batch_norm = True
    model_mode = 'ResNet34'
    num_classes = 10

    # nested dropout options
    apply_nested_dropout = False
    dropout_depth = 1
    p = 0.1
    seq_bound = 2 ** 7
    tol = 1e-3

    dataloader = imagenette.get_dataloader(train=True, batch_size=batch_size, normalize=normalize_data,
                                           image_mode=image_mode)
    testloader = imagenette.get_dataloader(train=False, batch_size=batch_size, normalize=normalize_data,
                                           image_mode=image_mode)
    model_kwargs = dict(num_classes=num_classes, mode=model_mode, activation=activation,
                        loss_criterion=loss_criterion, optim_alg=optim_alg, learning_rate=learning_rate,
                        patience=patience, batch_norm=batch_norm, dataset=dataset, image_mode=image_mode,
                        channels=channels, normalize_data=normalize_data, plateau_limit=plateau_limit,
                        apply_nested_dropout=apply_nested_dropout)
    if apply_nested_dropout:
        model_kwargs.update(dropout_depth=dropout_depth, p=p, sequence_bound=seq_bound, tol=tol)
    model = Classifier(**model_kwargs)
    optimizer = getattr(optim, optim_alg)(params=model.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=patience)

    current_time = utils.get_current_time()
    model_name = f'classifier-{model_mode}-{type(model).__name__}_{current_time}'
    model_dir = f'{utils.save_dir}/{model_name}'

    return training.train(model, optimizer, dataloader, testloader=testloader, model_dir=model_dir,
                          reconstruct=False, lr_scheduler=lr_scheduler, epochs=epochs, **model_kwargs)
