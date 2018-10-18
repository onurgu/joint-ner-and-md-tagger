#!/usr/bin/env python

from __future__ import print_function
import logging
import random
import sys
import time

from functools import partial

import os

import numpy as np

from utils.evaluation import eval_with_specific_model
from utils.loader import prepare_datasets

from toolkit.joint_ner_and_md_model import MainTaggerModel
from utils import models_path, eval_script, eval_logs_dir, read_parameters_from_sys_argv

from utils.dynetsaver import DynetSaver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


def train(sys_argv):

    # Read parameters from command line (skipping the program name, and the two others, i.e. --command train.
    opts, parameters = read_parameters_from_sys_argv(sys_argv)

    # Check evaluation script / folders
    if not os.path.isfile(eval_script):
        raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)

    # Reload
    if opts.model_epoch_path:
        model = MainTaggerModel(models_path=models_path,
                                model_path=opts.model_path,
                                model_epoch_dir_path=opts.model_epoch_path,
                                overwrite_mappings=opts.overwrite_mappings)
    else:
        # Initialize model
        model = MainTaggerModel(opts=opts,
                                parameters=parameters,
                                models_path=models_path, overwrite_mappings=opts.overwrite_mappings)

    print("MainTaggerModel location: {}".format(model.model_path))

    # Prepare the data
    # dev_data, _, \
    # id_to_tag, tag_scheme, test_data, \
    # train_data, train_stats, word_to_id, \
    # yuret_test_data, yuret_train_data = prepare_datasets(model, opts, parameters)

    data_dict, id_to_tag, word_to_id, stats_dict = prepare_datasets(model, opts, parameters)

    batch_size = opts.batch_size

    # Build the model
    model.build(training=True, **parameters)

    ### At this point, the training data is encoded in our format.

    #
    # Train network
    #

    n_epochs = opts.maximum_epochs  # number of epochs over the training set

    best_dev = -np.inf
    best_test = -np.inf

    if model.parameters['active_models'] in [1, 2, 3]:
        best_morph_dev = -np.inf
        best_morph_test = -np.inf

    model.trainer.set_clip_threshold(5.0)

    def update_loss(sentences_in_the_batch, loss_function):

        loss = loss_function(sentences_in_the_batch)
        loss.backward()
        model.trainer.update()
        if loss.value() / batch_size >= (10000000000.0 - 1):
            logging.error("BEEP")

        return loss.value()

    for epoch_no in range(n_epochs):
        start_time = time.time()
        epoch_costs = []
        print("Starting epoch {}...".format(epoch_no))

        n_samples_trained = 0

        loss_configuration_parameters = {}

        train_data = []
        for label in ["ner", "md"]:
            for purpose in ["train"]:
                train_data += data_dict[label][purpose]

        shuffled_data = list(train_data)
        random.shuffle(shuffled_data)

        index = 0
        while index < len(shuffled_data):
            batch_data = shuffled_data[index:(index + batch_size)]
            epoch_costs += [update_loss(batch_data,
                            loss_function=partial(model.get_loss,
                                                  loss_configuration_parameters=loss_configuration_parameters))]
            n_samples_trained += batch_size
            index += batch_size

            if n_samples_trained % 50 == 0 and n_samples_trained != 0:
                sys.stdout.write("%s%f " % ("G", np.mean(epoch_costs[-50:])))
                sys.stdout.flush()
                if np.mean(epoch_costs[-50:]) > 100:
                    logging.error("BEEP")

        print("")
        print("Avg. loss over training set: {epoch_loss_mean}".format(epoch_loss_mean=np.mean(epoch_costs)))

        model.trainer.status()

        # datasets_to_be_tested = {"ner": {"dev": data_dict["ner"]["dev"], "test": data_dict["ner"]["test"]},
        #                          "md": {"dev": data_dict["md"]["dev"], "test": data_dict["md"]["test"]}}

        datasets_to_be_tested = {label: {purpose: data_dict[label][purpose]
                                         for purpose in ["dev", "test"] if purpose in data_dict[label]}
                                 for label in ["ner", "md"]}

        f_scores, morph_accuracies, _ = eval_with_specific_model(model,
                                                                 epoch_no,
                                                                 datasets_to_be_tested,
                                                                 return_datasets_with_predicted_labels=False)

        if model.parameters['active_models'] in [0, 2, 3]:
            if "dev" in f_scores["ner"]:
                if best_dev < f_scores["ner"]["dev"]:
                    print("NER Epoch: %d New best dev score => best_dev, best_test: %lf %lf" % (epoch_no + 1,
                                                                                                       f_scores["ner"]["dev"],
                                                                                                       f_scores["ner"]["test"]))
                    best_dev = f_scores["ner"]["dev"]
                    best_test = f_scores["ner"]["test"]
                    model.save(epoch_no)
                    model.save_best_performances_and_costs(epoch_no,
                                                           best_performances=[f_scores["ner"]["dev"], f_scores["ner"]["test"]],
                                                           epoch_costs=epoch_costs)
                else:
                    print("NER Epoch: %d Best dev and accompanying test score, best_dev, best_test: %lf %lf" % (epoch_no + 1,
                                                                                                           best_dev,
                                                                                                           best_test))

        if model.parameters['active_models'] in [1, 2, 3]:
            if "dev" in morph_accuracies["md"]:
                if best_morph_dev < morph_accuracies["md"]["dev"]:
                    print("MORPH Epoch: %d New best dev score => best_dev, best_test: %lf %lf" %
                          (epoch_no, morph_accuracies["md"]["dev"], morph_accuracies["md"]["test"]))
                    best_morph_dev = morph_accuracies["md"]["dev"]
                    best_morph_test = morph_accuracies["md"]["test"]
                else:
                    print("MORPH Epoch: %d Best dev and accompanying test score, best_dev, best_test: %lf %lf"
                          % (epoch_no, best_morph_dev, best_morph_test))

        print("Epoch {} done. Average cost: {}".format(epoch_no, np.mean(epoch_costs)))
        print("MainTaggerModel dir: {}".format(model.model_path))
        print("Training took {} seconds for this epoch".format(time.time()-start_time))


if __name__ == "__main__":
    train(sys.argv)
