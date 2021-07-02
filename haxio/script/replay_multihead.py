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

import torch
import torch.optim.lr_scheduler
from torch.nn import CrossEntropyLoss

from avalanche.benchmarks import nc_benchmark
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.strategies import Naive
from haxio.models.multihead import resnet18
from haxio.script.utils import HaxioDataset
from haxio.utils import colored_print


def main(args):
    # --- CONFIG
    device = torch.device(f"cuda")
    # ---------

    train_set, val_set = HaxioDataset("/home/jizong/Workspace/avalanche/haxio/.data/medxl_v2", train_aug=True)
    scenario = nc_benchmark(train_set, val_set, n_experiences=3, task_labels=True, shuffle=False,
                            class_ids_from_zero_in_each_exp=True)
    # ---------

    # MODEL CREATION
    model = resnet18(input_dim=3)

    train_stream = scenario.train_stream
    test_stream = scenario.test_stream

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=False, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        # forgetting_metrics(experience=True),
        loggers=[interactive_logger])

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = Naive(model, torch.optim.Adam(model.parameters(), lr=0.00001),
                        CrossEntropyLoss(),
                        train_mb_size=100, train_epochs=20, eval_mb_size=100,
                        device=device,
                        plugins=[ReplayPlugin(mem_size=5)],
                        evaluator=eval_plugin
                        )
    cl_strategy.set_num_samplers_per_epoch(10000)

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
