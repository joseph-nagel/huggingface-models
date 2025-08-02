'''
Transfer learning.

Example
-------
python scripts/train_transfer_learning.py --tiny --max-epochs 3 --log-every-n-steps 1

'''

from argparse import ArgumentParser
from pathlib import Path

import mlflow
import torch
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.loggers import TensorBoardLogger, MLFlowLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
    StochasticWeightAveraging
)

from hf_utils import CIFAR10DataModule, LightningImgClassif


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--random-seed', type=int, required=False, help='Random seed')

    parser.add_argument('--ckpt-file', type=Path, required=False, help='Checkpoint for resuming')

    parser.add_argument('--logger', type=str, default='tensorboard', help='Logger')
    parser.add_argument('--save-dir', type=Path, default='run/', help='Save dir')
    parser.add_argument('--name', type=str, default='transfer', help='Experiment name')
    parser.add_argument('--version', type=str, required=False, help='Experiment version')

    parser.add_argument('--log-every-n-steps', type=int, default=50, help='How often to log train steps')
    parser.add_argument('--check-val-every-n-epoch', type=int, default=1, help='How many epochs between val. checks')

    parser.add_argument('--save-top-k', type=int, default=1, help='Number of best models to save')
    parser.add_argument('--save-every-n-epochs', type=int, default=1, help='Regular checkpointing interval')

    parser.add_argument('--data-dir', type=Path, default='run/data/', help='Data dir')

    parser.add_argument('--tiny', dest='tiny', action='store_true', help='Use tiny dataset')
    parser.add_argument('--no-tiny', dest='tiny', action='store_false', help='Use normal dataset')
    parser.set_defaults(tiny=False)

    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers')

    parser.add_argument('--model-name', type=str, default='microsoft/resnet-18', help='Pretrained model name')
    parser.add_argument('--num-labels', type=int, default=10, help='Number of target labels')

    parser.add_argument('--lr', type=float, default=1e-04, help='Initial learning rate')
    parser.add_argument('--lr-schedule', type=str, default='constant', choices=['constant', 'cosine'], help='LR schedule type')
    parser.add_argument('--lr-interval', type=str, default='epoch', choices=['epoch', 'step'], help='LR update interval')
    parser.add_argument('--lr-warmup', type=int, default=0, help='Warmup steps/epochs')

    parser.add_argument('--max-epochs', type=int, default=20, help='Max. number of training epochs')

    parser.add_argument('--patience', type=int, default=0, help='Early stopping patience')

    parser.add_argument('--swa-lrs', type=float, default=0.0, help='SWA learning rate')
    parser.add_argument('--swa-epoch-start', type=float, default=0.7, help='SWA start epoch')
    parser.add_argument('--annealing-epochs', type=int, default=10, help='SWA annealing epochs')
    parser.add_argument('--annealing-strategy', type=str, default='cos', help='SWA annealing strategy')

    parser.add_argument('--gradient-clip-val', type=float, default=0.0, help='Gradient clipping value')
    parser.add_argument('--gradient-clip-algorithm', type=str, default='norm', help='Gradient clipping mode')

    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--cpu', dest='gpu', action='store_false', help='Do not use GPU')
    parser.set_defaults(gpu=True)

    args = parser.parse_args()

    return args


def main(args):

    # set random seeds
    if args.random_seed is not None:
        _ = seed_everything(
            args.random_seed,
            workers=args.num_workers > 0
        )

    # initialize datamodule
    datamodule = CIFAR10DataModule(
        data_dir=args.data_dir,
        img_size=224,
        img_mean=(0.5, 0.5, 0.5),
        img_std=(0.5, 0.5, 0.5),
        random_state=42,
        tiny=args.tiny,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # initialize model
    model = LightningImgClassif(
        model_name=args.model_name,
        data_dir=args.data_dir,
        num_labels=args.num_labels,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        lr_interval=args.lr_interval,
        lr_warmup=args.lr_warmup
    )

    # set accelerator
    if args.gpu:
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    else:
        accelerator = 'cpu'

    # create logger
    if args.logger == 'tensorboard':
        logger = TensorBoardLogger(
            save_dir=args.save_dir,
            name=args.name,
            version=args.version
        )
    elif args.logger == 'mlflow':
        logger = MLFlowLogger(
            experiment_name=args.name,
            run_name=args.version,
            save_dir=args.save_dir / 'mlruns',
            log_model=True
        )
    else:
        raise ValueError(f'Unknown logger: {args.logger}')

    # set up LR monitor
    lr_monitor = LearningRateMonitor(logging_interval=None)

    callbacks = [lr_monitor]

    # set up checkpointing
    save_top_ckpt = ModelCheckpoint(
        filename='best',
        monitor='val_loss',
        mode='min',
        save_top_k=args.save_top_k
    )

    save_every_ckpt = ModelCheckpoint(
        filename='{epoch}',
        save_top_k=-1,
        every_n_epochs=args.save_every_n_epochs,
        save_last=True
    )

    callbacks.extend([save_top_ckpt, save_every_ckpt])

    # set up early stopping
    if args.patience > 0:
        early_stopping = EarlyStopping('val_loss', patience=args.patience)
        callbacks.append(early_stopping)

    # set up weight averaging
    if args.swa_lrs > 0:
        swa = StochasticWeightAveraging(
            swa_lrs=args.swa_lrs,
            swa_epoch_start=args.swa_epoch_start,
            annealing_epochs=args.annealing_epochs,
            annealing_strategy=args.annealing_strategy
        )
        callbacks.append(swa)

    # set up gradient clipping
    if args.gradient_clip_val > 0:
        gradient_clip_val = args.gradient_clip_val
        gradient_clip_algorithm = args.gradient_clip_algorithm
    else:
        gradient_clip_val = None
        gradient_clip_algorithm = None

    # initialize trainer
    trainer = Trainer(
        accelerator=accelerator,
        logger=logger,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
        deterministic=args.random_seed is not None
    )

    # start MLflow autologging
    save_dir = args.save_dir / 'mlruns'

    mlflow.set_tracking_uri(f'file:{save_dir}')
    mlflow.set_experiment(experiment_name=args.name)

    # TODO: disable logging and checkpointing other than mlflow autolog
    mlflow.pytorch.autolog(
        log_every_n_epoch=1,
        log_every_n_step=None,
        checkpoint_save_best_only=False,
        checkpoint_save_freq='epoch'
    )

    with mlflow.start_run(run_name=args.version):

        # check validation loss
        trainer.validate(
            model=model,
            datamodule=datamodule,
            ckpt_path=args.ckpt_file,
            verbose=False
        )

        # train model
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=args.ckpt_file
        )


if __name__ == '__main__':

    args = parse_args()
    main(args)

