import argparse

import torch
from torch.optim import Adam

from avalanche.benchmarks import nc_benchmark
from avalanche.evaluation.metrics import forgetting_metrics, \
    accuracy_metrics, loss_metrics, bwt_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import EWC
from haxio.models import resnet18
from haxio.script.utils import HaxioDataset
from haxio.utils import colored_print

"""
This example tests EWC on Split MNIST and Permuted MNIST.
It is possible to choose, among other options, between EWC with separate
penalties and online EWC with a single penalty.

On Permuted MNIST EWC maintains a very good performance on previous tasks
with a wide range of configurations. The average accuracy on previous tasks
at the end of training on all task is around 85%,
with a comparable training accuracy.

On Split MNIST, on the contrary, EWC is not able to remember previous tasks and
is subjected to complete forgetting in all configurations. The training accuracy
is above 90% but the average accuracy on previou tasks is around 20%.
"""


def main(args):
    model = resnet18(input_dim=3, num_classes=6)

    optimizer = Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device(f"cuda")

    train_set, val_set = HaxioDataset("/home/jizong/Workspace/avalanche/haxio/.data/medxl_v2", train_aug=True)

    scenario = nc_benchmark(
        train_set, val_set, 3, task_labels=False, shuffle=False)

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(
            minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        bwt_metrics(experience=True, stream=True),
        loggers=[interactive_logger])

    # create strategy
    strategy = EWC(model=model, optimizer=optimizer, criterion=criterion, mode=args.ewc_mode,
                   train_mb_size=128, train_epochs=20, eval_mb_size=128, device=device,
                   evaluator=eval_plugin, ewc_lambda=args.ewc)
    strategy.set_num_samplers_per_epoch(10000)

    # train on the selected scenario with the chosen strategy
    print('Starting experiment...')
    results = []
    for i, experience in enumerate(scenario.train_stream):
        print("Start training on experience ", experience.current_experience)

        strategy.train(experience)
        print("End training on experience", experience.current_experience)
        with colored_print():
            print('Computing accuracy on the test set')
            results.append(strategy.eval(scenario.test_stream[:i + 1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ewc_mode', type=str, choices=['separate', 'online'],
                        default='separate',
                        help='Choose between EWC and online.')
    parser.add_argument('--ewc_lambda', type=float, default=0.4,
                        help='Penalty hyperparameter for EWC')

    args = parser.parse_args()

    main(args)
