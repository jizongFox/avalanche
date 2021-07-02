################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-12-2020                                                             #
# Author(s): Andrea Cossu                                                      #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
"""
This example trains a Multi-head model on Split MNIST with Elastich Weight
Consolidation. Each experience has a different task label, which is used at test
time to select the appropriate head.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from avalanche.benchmarks import nc_benchmark
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import EWC
from haxio.models.multihead import resnet18
from haxio.script.utils import HaxioDataset, dataset_descript
from haxio.utils import colored_print


def main(args):
    # Config
    device = torch.device(f"cuda")
    # model
    model = resnet18(input_dim=3)

    # CL Benchmark Creation
    train_set, val_set = HaxioDataset("/home/jizong/Workspace/avalanche/haxio/.data/medxl_v2", train_aug=True)

    scenario = nc_benchmark(train_set, val_set, n_experiences=3, task_labels=True, shuffle=False,
                            class_ids_from_zero_in_each_exp=True)
    train_stream = scenario.train_stream
    test_stream = scenario.test_stream

    # Prepare for training & testing
    optimizer = Adam(model.parameters(), lr=0.0001)
    criterion = CrossEntropyLoss()

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=False, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger])

    # Choose a CL strategy
    strategy = EWC(
        model=model, optimizer=optimizer, criterion=criterion, mode=args.ewc_mode,
        train_mb_size=128, train_epochs=20, eval_mb_size=128, device=device,
        evaluator=eval_plugin, ewc_lambda=args.ewc_lambda)

    # train and test loop
    for i, train_task in enumerate(train_stream):
        dataset_descript(train_task)
        strategy.train(train_task)
        with colored_print():
            strategy.eval(test_stream[:i + 1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ewc_lambda', type=float, default=0.4, help='ewc_lambda:float')
    parser.add_argument('--ewc_mode', type=str, choices=['separate', 'online'],
                        default='separate',
                        help='Choose between EWC and online.')
    args = parser.parse_args()
    main(args)
