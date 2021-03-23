from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import shutil
import sys
from itertools import islice
from pathlib import Path
from random import sample
from typing import List, Text

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import webdataset as wds
from botocore.exceptions import ClientError
from PIL import Image
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics import Accuracy
from sklearn.model_selection import train_test_split
from torch import nn
from torch.multiprocessing import Queue
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import models, transforms


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        """
        Initialization of inherited lightning data module
        """
        super(CIFAR10DataModule, self).__init__()

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.valid_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                self.normalize,
            ]
        )

        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ]
        )
        self.args = kwargs

    def prepare_data(self):
        """
        Implementation of abstract class
        """

    @staticmethod
    def getNumFiles(input_path):
        return len(os.listdir(input_path)) - 1

    def setup(self, stage=None):
        """
        Downloads the data, parse it and split the data into train, test, validation data

        :param stage: Stage - training or testing
        """

        data_path = self.args["train_glob"]

        train_base_url = data_path + "/train"
        val_base_url = data_path + "/val"
        test_base_url = data_path + "/test"

        train_count = self.getNumFiles(train_base_url)
        val_count = self.getNumFiles(val_base_url)
        test_count = self.getNumFiles(test_base_url)

        train_url = "{}/{}-{}".format(
            train_base_url, "train", "{0.." + str(train_count) + "}.tar"
        )
        valid_url = "{}/{}-{}".format(
            val_base_url, "val", "{0.." + str(val_count) + "}.tar"
        )
        test_url = "{}/{}-{}".format(
            test_base_url, "test", "{0.." + str(test_count) + "}.tar"
        )

        self.train_dataset = (
            wds.Dataset(train_url, handler=wds.warn_and_continue, length=40000 // 40)
            .shuffle(100)
            .decode("pil")
            .rename(image="ppm;jpg;jpeg;png", info="cls")
            .map_dict(image=self.train_transform)
            .to_tuple("image", "info")
            .batched(40)
        )

        self.valid_dataset = (
            wds.Dataset(valid_url, handler=wds.warn_and_continue, length=10000 // 20)
            .shuffle(100)
            .decode("pil")
            .rename(image="ppm", info="cls")
            .map_dict(image=self.valid_transform)
            .to_tuple("image", "info")
            .batched(20)
        )

        self.test_dataset = (
            wds.Dataset(test_url, handler=wds.warn_and_continue, length=10000 // 20)
            .shuffle(100)
            .decode("pil")
            .rename(image="ppm", info="cls")
            .map_dict(image=self.valid_transform)
            .to_tuple("image", "info")
            .batched(20)
        )

    def create_data_loader(self, dataset, batch_size, num_workers):
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        self.train_data_loader = self.create_data_loader(
            self.train_dataset,
            self.args["train_batch_size"],
            self.args["train_num_workers"],
        )
        return self.train_data_loader

    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        self.val_data_loader = self.create_data_loader(
            self.valid_dataset,
            self.args["val_batch_size"],
            self.args["val_num_workers"],
        )
        return self.val_data_loader

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        self.test_data_loader = self.create_data_loader(
            self.test_dataset, self.args["val_batch_size"], self.args["val_num_workers"]
        )
        return self.test_data_loader


class CIFAR10Classifier(pl.LightningModule):
    def __init__(self, **kwargs):
        """
        Initializes the network, optimizer and scheduler
        """
        super(CIFAR10Classifier, self).__init__()
        self.model_conv = models.resnet50(pretrained=True)
        for param in self.model_conv.parameters():
            param.requires_grad = False
        num_ftrs = self.model_conv.fc.in_features
        num_classes = 10
        self.model_conv.fc = nn.Linear(num_ftrs, num_classes)

        self.scheduler = None
        self.optimizer = None
        self.args = kwargs

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x):
        out = self.model_conv(x)
        return out

    def training_step(self, train_batch, batch_idx):
        if batch_idx == 0:
            self.reference_image = (train_batch[0][0]).unsqueeze(0)
            #self.reference_image.resize((1,1,28,28))
            print("\n\nREFERENCE IMAGE!!!")
            print(self.reference_image.shape)
        x, y = train_batch
        output = self.forward(x)
        _, y_hat = torch.max(output, dim=1)
        loss = F.cross_entropy(output, y)
        self.log("train_loss", loss)
        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc.compute())
        return {"loss": loss}

    def test_step(self, test_batch, batch_idx):

        x, y = test_batch
        output = self.forward(x)
        _, y_hat = torch.max(output, dim=1)
        loss = F.cross_entropy(output, y)
        if self.args["accelerator"] is not None:
            self.log("test_loss", loss, sync_dist=True)
        else:
            self.log("test_loss", loss)
        self.test_acc(y_hat, y)
        self.log("test_acc", self.test_acc.compute())
        return {"test_acc": self.test_acc.compute()}

    def validation_step(self, val_batch, batch_idx):

        x, y = val_batch
        output = self.forward(x)
        _, y_hat = torch.max(output, dim=1)
        loss = F.cross_entropy(output, y)
        if self.args["accelerator"] is not None:
            self.log("val_loss", loss, sync_dist=True)
        else:
            self.log("val_loss", loss)
        self.val_acc(y_hat, y)
        self.log("val_acc", self.val_acc.compute())
        return {"val_step_loss": loss, "val_loss": loss}

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args["lr"])
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "val_loss",
        }
        return [self.optimizer], [self.scheduler]

    def makegrid(self, output, numrows):
        outer = torch.Tensor.cpu(output).detach()
        plt.figure(figsize=(20, 5))
        b = np.array([]).reshape(0, outer.shape[2])
        c = np.array([]).reshape(numrows * outer.shape[2], 0)
        i = 0
        j = 0
        while i < outer.shape[1]:
            img = outer[0][i]
            b = np.concatenate((img, b), axis=0)
            j += 1
            if j == numrows:
                c = np.concatenate((c, b), axis=1)
                b = np.array([]).reshape(0, outer.shape[2])
                j = 0

            i += 1
        return c

    def showActivations(self, x):

        # logging reference image
        self.logger.experiment.add_image(
            "input", torch.Tensor.cpu(x[0][0]), self.current_epoch, dataformats="HW"
        )

        # logging layer 1 activations
        out = self.model_conv.conv1(x)
        c = self.makegrid(out, 4)
        self.logger.experiment.add_image(
            "layer 1", c, self.current_epoch, dataformats="HW"
        )

    def training_epoch_end(self, outputs):
        self.showActivations(self.reference_image)

        # Logging graph
        if(self.current_epoch==0):
            sampleImg=torch.rand((1,3,64,64))
            self.logger.experiment.add_graph(CIFAR10Classifier(),sampleImg)


def preprocess_fn(inputs):

    print("Preprocess function inputs...")
    print(inputs)

    # Path(output_path + "/train").mkdir(parents=True, exist_ok=True)
    # Path(output_path + "/val").mkdir(parents=True, exist_ok=True)
    # Path(output_path + "/test").mkdir(parents=True, exist_ok=True)

    # random_seed = 25
    # y = trainset.targets
    # trainset, valset, y_train, y_val = train_test_split(
    #     trainset, y, stratify=y, shuffle=True, test_size=0.2, random_state=random_seed
    # )

    # for name in [(trainset, "train"), (valset, "val"), (testset, "test")]:
    #     with wds.ShardWriter(
    #         output_path + "/" + str(name[1]) + "/" +  str(name[1]) + "-%d.tar", maxcount=1000
    #     ) as sink:
    #         for index, (image, cls) in enumerate(name[0]):
    #             sink.write({"__key__": "%06d" % index, "ppm": image, "cls": cls})