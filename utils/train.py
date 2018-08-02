#!/usr/bin/env python

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

    print "MainTaggerModel location: %s" % model.model_path

    # Prepare the data
    dev_data, _, \
    id_to_tag, tag_scheme, test_data, \
    train_data, train_stats, word_to_id, \
    yuret_test_data, yuret_train_data = prepare_datasets(model, opts, parameters)

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

    def get_loss_for_a_batch(batch_data,
                             loss_function=partial(model.get_loss, gungor_data=True),
                             label="G"):

        loss_value = update_loss(batch_data, loss_function)

        return loss_value

    def update_loss(sentences_in_the_batch, loss_function):

        loss = loss_function(sentences_in_the_batch)
        loss.backward()
        model.trainer.update()
        if loss.value() / batch_size >= (10000000000.0 - 1):
            logging.error("BEEP")

        return loss.value()

    for epoch in range(n_epochs):
        start_time = time.time()
        epoch_costs = []
        print "Starting epoch %i..." % epoch

        count = 0
        yuret_count = 0

        if opts.use_buckets:
            pass
            # permuted_bucket_ids = np.random.permutation(range(len(train_buckets)))
            #
            # for bucket_id in list(permuted_bucket_ids):
            #     bucket_data = train_buckets[bucket_id]
            #
            #     print "bucket_id: %d, len(batch_data): %d" % (bucket_id, len(batch_data))
            #
            #     shuffled_data = list(bucket_data)
            #     random.shuffle(shuffled_data)
            #
            #     index = 0
            #     while index < len(shuffled_data):
            #         batch_data = shuffled_data[index:(index + batch_size)]
            #         epoch_costs += [get_loss_for_a_batch(batch_data)]
            #         count += batch_size
            #         index += batch_size
            #
            #         if count % 50 == 0 and count != 0:
            #             sys.stdout.write("%s%f " % ("G", np.mean(epoch_costs[-50:])))
            #             sys.stdout.flush()
            #             if np.mean(epoch_costs[-50:]) > 100:
            #                 logging.error("BEEP")
            #     print ""
        else:
            shuffled_data = list(train_data)
            random.shuffle(shuffled_data)

            index = 0
            while index < len(shuffled_data):
                batch_data = shuffled_data[index:(index + batch_size)]
                epoch_costs += [get_loss_for_a_batch(batch_data)]
                count += batch_size
                index += batch_size

                if count % 50 == 0 and count != 0:
                    sys.stdout.write("%s%f " % ("G", np.mean(epoch_costs[-50:])))
                    sys.stdout.flush()
                    if np.mean(epoch_costs[-50:]) > 100:
                        logging.error("BEEP")

        print ""

        if model.parameters["train_with_yuret"]:
            shuffled_data = list(yuret_train_data)
            random.shuffle(shuffled_data)

            index = 0
            while index < len(shuffled_data):
                batch_data = shuffled_data[index:(index + batch_size)]
                epoch_costs += [get_loss_for_a_batch(batch_data,
                                                    loss_function=partial(model.get_loss,
                                                                          gungor_data=False),
                                                    label="Y")]
                count += batch_size
                index += batch_size

                if count % 50 == 0 and count != 0:
                    sys.stdout.write("%s%f " % ("Y", np.mean(epoch_costs[-50:])))
                    sys.stdout.flush()
                    if np.mean(epoch_costs[-50:]) > 100:
                        logging.error("BEEP")

            print ""

        model.trainer.status()

        datasets_to_be_tested = [("dev", dev_data),
                                ("test", test_data)]
        if model.parameters['test_with_yuret']:
            datasets_to_be_tested.append(("yuret", yuret_test_data))

        f_scores, morph_accuracies, _ = eval_with_specific_model(model,
                                                                 epoch,
                                                                 datasets_to_be_tested,
                                                                 return_datasets_with_predicted_labels=False)

        if model.parameters['active_models'] in [0, 2, 3]:
            if best_dev < f_scores["dev"]:
                print("NER Epoch: %d New best dev score => best_dev, best_test: %lf %lf" % (epoch + 1,
                                                                                                   f_scores["dev"],
                                                                                                   f_scores["test"]))
                best_dev = f_scores["dev"]
                best_test = f_scores["test"]
                model.save(epoch)
                model.save_best_performances_and_costs(epoch,
                                                       best_performances=[f_scores["dev"], f_scores["test"]],
                                                       epoch_costs=epoch_costs)
            else:
                print("NER Epoch: %d Best dev and accompanying test score, best_dev, best_test: %lf %lf" % (epoch + 1,
                                                                                                           best_dev,
                                                                                                           best_test))

        if model.parameters['active_models'] in [1, 2, 3]:
            if best_morph_dev < morph_accuracies["dev"]:
                print("MORPH Epoch: %d New best dev score => best_dev, best_test: %lf %lf" %
                      (epoch, morph_accuracies["dev"], morph_accuracies["test"]))
                best_morph_dev = morph_accuracies["dev"]
                best_morph_test = morph_accuracies["test"]
                if parameters['test_with_yuret']:
                    best_morph_yuret = morph_accuracies["yuret"]
                    print("YURET Epoch: %d New best dev score => best_dev, best_test: %lf %lf" %
                          (epoch, 0.0, morph_accuracies["yuret"]))
                    # we do not save in this case, just reporting
            else:
                print("MORPH Epoch: %d Best dev and accompanying test score, best_dev, best_test: %lf %lf"
                      % (epoch, best_morph_dev, best_morph_test))
                if parameters['test_with_yuret']:
                    print("YURET Epoch: %d Best dev and accompanying test score, best_dev, best_test: %lf %lf"
                          % (epoch, 0.0, best_morph_yuret))

        print "Epoch %i done. Average cost: %f" % (epoch, np.mean(epoch_costs))
        print "MainTaggerModel dir: %s" % model.model_path
        print "Training took %lf seconds for this epoch" % (time.time()-start_time)

if __name__ == "__main__":
    train(sys.argv)
