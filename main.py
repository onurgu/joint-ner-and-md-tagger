
import argparse

from utils.train import train
from utils.evaluation import evaluate

import sys

if __name__ == "__main__":

    backup_sys_argv = list(sys.argv)

    parser = argparse.ArgumentParser("joint ner and md tagger")

    parser.add_argument("--command", default="train", choices=["train", "evaluate", "predict", "webserver"])

    args = parser.parse_args(backup_sys_argv[1:3])

    sys_argv_to_be_transferred = [backup_sys_argv[0]] + backup_sys_argv[3:]
    if args.command == "train":
        train(sys_argv_to_be_transferred)
    elif args.command == "evaluate":
        evaluate(sys_argv_to_be_transferred)
