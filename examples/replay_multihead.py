################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 12-10-2020                                                             #
# Author(s): Vincenzo Lomonaco                                                 #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This is a simple example on how to use the Replay strategy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from os.path import expanduser

import torch
import torch.optim.lr_scheduler
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, RandomCrop

from avalanche.benchmarks import nc_benchmark
from avalanche.evaluation.metrics import forgetting_metrics, \
    accuracy_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.strategies import Naive
from haxio.models.multihead import resnet18
from haxio.utils import colored_print


def main(args):
    # --- CONFIG
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                             args.cuda >= 0 else "cpu")
    n_batches = 5
    # ---------

    # --- TRANSFORMATIONS
    train_transform = transforms.Compose([
        RandomCrop(28, padding=4),
        ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # ---------

    # --- SCENARIO CREATION
    mnist_train = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                        train=True, download=True, transform=train_transform)
    mnist_test = MNIST(root=expanduser("~") + "/.avalanche/data/mnist/",
                       train=False, download=True, transform=test_transform)
    scenario = nc_benchmark(
        mnist_train, mnist_test, n_batches, task_labels=True, seed=1234, class_ids_from_zero_in_each_exp=True)
    # ---------

    # MODEL CREATION
    model = resnet18(input_dim=1)

    train_stream = scenario.train_stream
    test_stream = scenario.test_stream

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True),
        # loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger])

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = Naive(model, torch.optim.Adam(model.parameters(), lr=0.001),
                        CrossEntropyLoss(),
                        train_mb_size=100, train_epochs=4, eval_mb_size=100,
                        device=device,
                        plugins=[ReplayPlugin(mem_size=100)],
                        evaluator=eval_plugin
                        )

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for i, experience in enumerate(train_stream):
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience)
        print('Training completed')
        print('Computing accuracy on the whole test set')
        with colored_print():
            results.append(cl_strategy.eval(test_stream[:i + 1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0,
                        help='Select zero-indexed cuda device. -1 to use CPU.')
    args = parser.parse_args()
    main(args)
