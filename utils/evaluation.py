"""Evaluation

"""
from __future__ import absolute_import
from __future__ import division

from collections import defaultdict as dd
import logging
import math
import sys

import subprocess

import codecs
import numpy as np

import os

import dynet

from toolkit.joint_ner_and_md_model import MainTaggerModel
from utils import eval_script, iobes_iob, eval_logs_dir
from utils.loader import prepare_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval")

def eval_with_specific_model(model, epoch, buckets_list, integration_mode, active_models,
                             *args): # FLAGS.eval_dir
    # type: (MainTaggerModel, int, list, object, object) -> object
    id_to_tag, batch_size, eval_dir, tag_scheme = args

    f_scores = {}
    dataset_labels = ["dev", "test", "yuret"]

    total_correct_disambs = {dataset_label: 0 for dataset_label in dataset_labels}
    total_disamb_targets = {dataset_label: 0 for dataset_label in dataset_labels}
    if active_models in [1, 2, 3]:
        detailed_correct_disambs = {dataset_label: dd(int) for dataset_label in dataset_labels}
        detailed_total_target_disambs = {dataset_label: dd(int) for dataset_label in dataset_labels}

    for dataset_label, dataset_as_list in buckets_list:

        if len(dataset_as_list) == 0:
            print "Skipping to evaluate %s dataset as it is empty" % dataset_label
            total_correct_disambs[dataset_label] = -1
            total_disamb_targets[dataset_label] = 1
            continue

        print "Starting to evaluate %s dataset" % dataset_label
        predictions = []
        n_tags = len(id_to_tag)
        count = np.zeros((n_tags, n_tags), dtype=np.int32)

        n_batches = int(math.ceil(float(len(dataset_as_list)) / batch_size))

        print "dataset_label: %s" % dataset_label
        print ("n_batches: %d" % n_batches)

        for batch_idx in range(n_batches):
            # print("batch_idx: %d" % batch_idx)
            sys.stdout.write(". ")
            sys.stdout.flush()

            sentences_in_the_batch = dataset_as_list[
                                     (batch_idx * batch_size):((batch_idx + 1) * batch_size)]

            for sentence in sentences_in_the_batch:
                dynet.renew_cg()

                sentence_length = len(sentence['word_ids'])

                if active_models in [2, 3]:
                    selected_morph_analyzes, decoded_tags = model.predict(sentence)
                elif active_models in [1]:
                    selected_morph_analyzes, _ = model.predict(sentence)
                elif active_models in [0]:
                    decoded_tags = model.predict(sentence)

                if active_models in [0, 2, 3]: # i.e. not only MD
                    p_tags = [id_to_tag[p_tag] for p_tag in decoded_tags]
                    r_tags = [id_to_tag[p_tag] for p_tag in sentence['tag_ids']]
                    if tag_scheme == 'iobes':
                        p_tags = iobes_iob(p_tags)
                        r_tags = iobes_iob(r_tags)

                    for i, (word_id, y_pred, y_real) in enumerate(
                            zip(sentence['word_ids'], decoded_tags,
                                sentence['tag_ids'])):
                        new_line = " ".join([sentence['str_words'][i]] + [r_tags[i], p_tags[i]])
                        predictions.append(new_line)
                        count[y_real, y_pred] += 1
                    predictions.append("")

                if active_models in [1, 2, 3]:
                    n_correct_morph_disambs = \
                        sum([x == y for x, y, z in zip(selected_morph_analyzes,
                                                    sentence['golden_morph_analysis_indices'],
                                                    sentence['morpho_analyzes_tags']) if len(z) > 1])
                    total_correct_disambs[dataset_label] += n_correct_morph_disambs
                    total_disamb_targets[dataset_label] += sum([1 for el in sentence['morpho_analyzes_tags'] if len(el) > 1])
                    for key, value in [(len(el), x == y) for el, x, y in zip(sentence['morpho_analyzes_tags'],
                                                           selected_morph_analyzes,
                                                           sentence['golden_morph_analysis_indices'])]:
                        if value:
                            detailed_correct_disambs[dataset_label][key] += 1
                        detailed_total_target_disambs[dataset_label][key] += 1
                    # total_possible_analyzes += sum([len(el) for el in sentence['morpho_analyzes_tags'] if len(el) > 1])

        print ""

        if active_models in [0, 2, 3]:
            # Write predictions to disk and run CoNLL script externally
            eval_id = np.random.randint(1000000, 2000000)
            output_path = os.path.join(eval_dir,
                                       "%s.eval.%i.epoch-%04d.output" % (
                                           dataset_label, eval_id, epoch))
            scores_path = os.path.join(eval_dir,
                                       "%s.eval.%i.epoch-%04d.scores" % (
                                           dataset_label, eval_id, epoch))
            with codecs.open(output_path, 'w', 'utf8') as f:
                f.write("\n".join(predictions))

            print "Evaluating the %s dataset with conlleval script" % dataset_label
            command_string = "%s < %s > %s" % (eval_script, output_path, scores_path)
            print command_string
            # os.system(command_string)
            # sys.exit(0)
            with codecs.open(output_path, "r", encoding="utf-8") as output_path_f:
                eval_lines = [x.rstrip() for x in subprocess.check_output([eval_script],
                                                                          stdin=output_path_f).split(
                    "\n")]

                # CoNLL evaluation results
                # eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
                for line in eval_lines:
                    print line
                f_scores[dataset_label] = float(eval_lines[1].split(" ")[-1])

        if active_models in [1, 2, 3]:
            for n_possible_analyzes in map(int, detailed_correct_disambs[dataset_label].keys()):
                print "%s %d %d/%d" % (dataset_label,
                                       n_possible_analyzes,
                                       detailed_correct_disambs[dataset_label][n_possible_analyzes],
                                       detailed_total_target_disambs[dataset_label][n_possible_analyzes])
    if active_models in [0]:
        return f_scores, {}
    else:
        result = {}
        for dataset_label in dataset_labels:
            if total_disamb_targets[dataset_label] == 0:
                total_correct_disambs[dataset_label] = -1
                total_disamb_targets[dataset_label] = 1
            result[dataset_label] = \
                total_correct_disambs[dataset_label] / float(total_disamb_targets[dataset_label])

        return f_scores, result


def evaluate_model_dir_path(models_dir_path, model_dir_path, model_epoch_dir_path):

    import os
    from utils import read_parameters_from_file

    parameters, opts = read_parameters_from_file(os.path.join(model_dir_path, "parameters.pkl"),
                                                 os.path.join(model_dir_path, "opts.pkl"))

    model = MainTaggerModel(models_path=models_dir_path,
                            model_path=model_dir_path,
                            model_epoch_dir_path=model_epoch_dir_path)

    # Prepare the data
    dev_data, dico_words_train, \
    id_to_tag, tag_scheme, test_data, \
    train_data, train_stats, word_to_id, \
    yuret_test_data, yuret_train_data = prepare_datasets(model, opts, parameters, for_training=False)

    batch_size = opts.batch_size

    # Build the model
    model.build(**parameters)

    datasets_to_be_tested = [("dev", dev_data),
                             ("test", test_data)]

    f_scores, morph_accuracies = eval_with_specific_model(model, -1, datasets_to_be_tested,
                                                          model.parameters['integration_mode'],
                                                          model.parameters['active_models'],
                                                          id_to_tag, batch_size,
                                                          eval_logs_dir,
                                                          tag_scheme)

    print f_scores
    print morph_accuracies


def evaluate(sys_argv):

    from utils import read_args

    opts = read_args(args_as_a_list=sys_argv[1:])

    from utils.train import models_path

    evaluate_model_dir_path(
        models_dir_path=models_path,
        model_dir_path=opts.model_path,
        model_epoch_dir_path=opts.model_epoch_path
    )