import os
import astroaugmentations as AA
from astroaugmentations.datasets.galaxy_mnist import GalaxyMNIST

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch import optim, nn, utils, Tensor
import torchvision.models as models
from torchmetrics.functional import accuracy
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

# Custom Logging of Predictions, Adapted from: https://github.com/wandb/examples/blob/master/colabs/pytorch-lightning/Optimize_Pytorch_Lightning_models_with_Weights_%26_Biases.ipynb
class LogPredictionsCallback(pl.Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = [img for img in x[:n]]
            captions = [f'Ground Truth: {y_i} - Prediction: {y_pred}' for y_i, y_pred in zip(y[:n], outputs[:n])]

            # Option 1: log images with `WandbLogger.log_image`
            wandb_logger.log_image(key='sample_images', images=images, caption=captions)

            # Option 2: log predictions as a Table
            #columns = ['image', 'ground truth', 'prediction']
            #data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
            #wandb_logger.log_table(key='sample_table', columns=columns, data=data)

class DataModule(pl.LightningDataModule):
    def __init__(self, transform, config, batch_size=64):
        super().__init__()
        self.transform = transform
        self.batch_size = batch_size
        self.config = config
        self.data_dir = self.config['data_dir']

    def prepare_data(self):
        # download
        GalaxyMNIST(self.data_dir, train=True, download=True)
        GalaxyMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = GalaxyMNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [6000, 2000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            #self.mnist_test = GalaxyMNIST(self.data_dir, train=False, transform=self.transform) # test on augmented images
            self.mnist_test = GalaxyMNIST(self.data_dir, train=False, transform=None) # test on unaugmented images

        if stage == "predict" or stage is None:
            #self.mnist_predict = GalaxyMNIST(self.data_dir, train=False, transform=self.transform) # predict on augmented images
            self.mnist_predict = GalaxyMNIST(self.data_dir, train=False, transform=None) # predict on unaugmented images

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=self.config['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.config['num_workers'])

    #def test_dataloader(self):
    #    return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=self.config['num_workers'])

# define the LightningModule
class Classifier(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        #self.save_hyperparameters(ignore=['model'])
        self.config = config

    def forward(self, x):
        x_tmp = x.permute(0,3,1,2).double()
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        # validation metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        x_tmp = x.permute(0,3,1,2).double()
        y_hat = self.model(x_tmp)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = accuracy(y_hat, y)
        return loss, acc

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])

def main():
    wandb.init()

    # Defien Transformations
    if wandb.config['transform'] == 'default':
        train_no_augmentation = GalaxyMNIST(
            root=wandb.config['data_dir'],
            train=True, download=True, transform=None
            )
        transform = A.Compose([
            A.Lambda(
                name='AA.composed.ImgOptical',
                image=AA.composed.ImgOptical(
                    dataset=train_no_augmentation
                ), always_apply=True
            )
        ])
    elif wandb.config['transform'] == 'computer_vision':
        transform = A.Compose([
            A.Affine(
                scale=(0.9, 1.2),
                rotate=(0, 360),
                interpolation=2,
                border_mode=0,
                value=0,
                p=1
            )
        ])
    elif wandb.config['transform'] == 'None':
        transform = None
    else:
        raise NotImplementedError

    # Model architecture
    if wandb.config['model']=='resnet50':
        model = models.resnet50()
        model.fc = nn.Linear(
            in_features=model.fc.in_features,
            out_features=wandb.config["num_classes"],
            bias=True
            )
    else:
        raise NotImplementedError

    classifier = Classifier(model, config=wandb.config)
    wandb_logger = WandbLogger(project=wandb.config['project'], job_type='train') ###
    trainer = pl.Trainer(
        limit_train_batches=None, max_epochs=wandb.config['epochs'],
        logger=wandb_logger,
        callbacks=[
            pl.callbacks.EarlyStopping("val_loss", min_delta=0.01, patience=8),
            #LogPredictionsCallback() # apparently not working for now
            ],
        # I think I am double logging as wandb is logging the model weights as well?
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val/loss',
            dirpath='./checkpoints',
            filename='galaxyMNIST-epoch{epoch:02d}-val_loss{val/loss:.2f}',
            auto_insert_metric_name=False,
            save_top_k=3,
            save_last=True
            )
        )
    trainer.fit(
        model=classifier.double(),
        train_dataloaders=DataModule(transform=transform, config=wandb.config)
        )


if __name__=="__main__":
    main()
