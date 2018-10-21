"""Evaluation

"""



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
from utils.loader import prepare_datasets, extract_mapping_dictionaries_from_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval")


def eval_with_specific_model(model,
                             epoch,
                             datasets_to_be_predicted,
                             return_datasets_with_predicted_labels=False):

    # type: (MainTaggerModel, int, dict, bool) -> object
    eval_dir = eval_logs_dir
    batch_size = 1 # model.parameters['batch_size'] TODO: switch back for new models.
    integration_mode = model.parameters['integration_mode']
    active_models = model.parameters['active_models']
    id_to_tag = model.id_to_tag
    tag_scheme = model.parameters['t_s']

    f_scores = {"ner":{}}
    # dataset_labels = ["dev", "test", "yuret"]
    # dataset_labels = [map(lambda purpose: label+"_"+purpose, datasets_to_be_predicted[label].keys())
    #                   for label in datasets_to_be_predicted.keys()]

    # total_correct_disambs = {dataset_label: 0 for dataset_label in dataset_labels}
    # total_disamb_targets = {dataset_label: 0 for dataset_label in dataset_labels}

    total_correct_disambs = {label: {purpose: 0
                                     for purpose in list(datasets_to_be_predicted[label].keys())}
                             for label in list(datasets_to_be_predicted.keys())}

    total_disamb_targets = {label: {purpose: 0
                                     for purpose in list(datasets_to_be_predicted[label].keys())}
                             for label in list(datasets_to_be_predicted.keys())}

    if active_models in [1, 2, 3]:

        detailed_correct_disambs = {label: {purpose: dd(int)
                                        for purpose in list(datasets_to_be_predicted[label].keys())}
                                for label in list(datasets_to_be_predicted.keys())}

        detailed_total_target_disambs = {label: {purpose: dd(int)
                                        for purpose in list(datasets_to_be_predicted[label].keys())}
                                for label in list(datasets_to_be_predicted.keys())}

    datasets_with_predicted_labels = {label: {purpose: {}
                                        for purpose in list(datasets_to_be_predicted[label].keys())}
                                for label in list(datasets_to_be_predicted.keys())}

    # for dataset_label, dataset_as_list in datasets_to_be_predicted:
    for label in list(datasets_to_be_predicted.keys()):
        for purpose in list(datasets_to_be_predicted[label].keys()):

            dataset_as_list = datasets_to_be_predicted[label][purpose]

            if len(dataset_as_list) == 0:
                print("Skipping to evaluate %s dataset as it is empty" % (label+"_"+purpose))
                total_correct_disambs[label][purpose] = -1
                total_disamb_targets[label][purpose] = 1
                continue

            print("Starting to evaluate %s dataset" % (label+"_"+purpose))
            predictions = []
            n_tags = len(id_to_tag)
            count = np.zeros((n_tags, n_tags), dtype=np.int32)

            n_batches = int(math.ceil(float(len(dataset_as_list)) / batch_size))

            print("dataset_label: %s" % (label+"_"+purpose))
            print(("n_batches: %d" % n_batches))

            for batch_idx in range(n_batches):
                # print("batch_idx: %d" % batch_idx)
                sys.stdout.write(". ")
                sys.stdout.flush()

                sentences_in_the_batch = dataset_as_list[
                                         (batch_idx * batch_size):((batch_idx + 1) * batch_size)]

                for sentence in sentences_in_the_batch:
                    dynet.renew_cg()

                    sentence_length = len(sentence['word_ids'])

                    if active_models in [2, 3] and label in "ner md".split(" "):
                        selected_morph_analyzes, decoded_tags = model.predict(sentence)
                    elif active_models in [1] and label == "md":
                        selected_morph_analyzes, _ = model.predict(sentence)
                    elif active_models in [0] and label == "ner":
                        _, decoded_tags = model.predict(sentence)

                    if active_models in [0, 2, 3] and label == "ner": # i.e. except MD
                        p_tags = [id_to_tag[p_tag] for p_tag in decoded_tags]
                        r_tags = [id_to_tag[r_tag] for r_tag in sentence['tag_ids']]
                        if tag_scheme == 'iobes':
                            p_tags = iobes_iob(p_tags)
                            r_tags = iobes_iob(r_tags)

                        for i, (y_pred, y_real) in enumerate(
                                zip(decoded_tags, sentence['tag_ids'])):
                            new_line = " ".join([sentence['str_words'][i]] + [r_tags[i], p_tags[i]])
                            predictions.append(new_line)
                            count[y_real, y_pred] += 1
                        predictions.append("")

                    if active_models in [1, 2, 3] and label == "md":
                        n_correct_morph_disambs = \
                            sum([x == y for x, y, z in zip(selected_morph_analyzes,
                                                        sentence['golden_morph_analysis_indices'],
                                                        sentence['morpho_analyzes_tags']) if len(z) > 1])
                        total_correct_disambs[label][purpose] += n_correct_morph_disambs
                        total_disamb_targets[label][purpose] += sum([1 for el in sentence['morpho_analyzes_tags'] if len(el) > 1])
                        for key, value in [(len(el), x == y) for el, x, y in zip(sentence['morpho_analyzes_tags'],
                                                               selected_morph_analyzes,
                                                               sentence['golden_morph_analysis_indices'])]:
                            if value:
                                detailed_correct_disambs[label][purpose][key] += 1
                            detailed_total_target_disambs[label][purpose][key] += 1
                        # total_possible_analyzes += sum([len(el) for el in sentence['morpho_analyzes_tags'] if len(el) > 1])

            print("")

            if active_models in [0, 2, 3] and label == "ner":
                # Write predictions to disk and run CoNLL script externally
                eval_id = np.random.randint(1000000, 2000000)
                output_path = os.path.join(eval_dir,
                                           "%s.eval.%i.epoch-%04d.output" % (
                                               (label + "_" + purpose), eval_id, epoch))
                scores_path = os.path.join(eval_dir,
                                           "%s.eval.%i.epoch-%04d.scores" % (
                                               (label + "_" + purpose), eval_id, epoch))
                with codecs.open(output_path, 'w', 'utf8') as f:
                    f.write("\n".join(predictions))

                print("Evaluating the %s dataset with conlleval script" % (label + "_" + purpose))
                command_string = "%s < %s > %s" % (eval_script, output_path, scores_path)
                print(command_string)
                # os.system(command_string)
                # sys.exit(0)
                with codecs.open(output_path, "r", encoding="utf-8") as output_path_f:
                    eval_lines = [x.rstrip() for x in subprocess.check_output([eval_script],
                                                                              stdin=output_path_f).decode("utf8").split(
                        "\n")]

                    # CoNLL evaluation results
                    # eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
                    for line in eval_lines:
                        print(line)
                    f_scores["ner"][purpose] = float(eval_lines[1].split(" ")[-1])

            if active_models in [1, 2, 3]:
                for n_possible_analyzes in map(int, list(detailed_correct_disambs[label][purpose].keys())):
                    print("%s %d %d/%d" % ((label + "_" + purpose),
                                           n_possible_analyzes,
                                           detailed_correct_disambs[label][purpose][n_possible_analyzes],
                                           detailed_total_target_disambs[label][purpose][n_possible_analyzes]))

            if return_datasets_with_predicted_labels:
                datasets_with_predicted_labels[label][purpose] = predictions

    disambiguation_accuracies = {label: {} for label in list(datasets_to_be_predicted.keys())}
    if active_models in [0]:
        pass
    else:
        for label in list(datasets_to_be_predicted.keys()):
            for purpose in list(datasets_to_be_predicted[label].keys()):
                if total_disamb_targets[label][purpose] == 0:
                    total_correct_disambs[label][purpose] = -1
                    total_disamb_targets[label][purpose] = 1
                disambiguation_accuracies[label][purpose] = \
                    total_correct_disambs[label][purpose] / float(total_disamb_targets[label][purpose])

    return f_scores, disambiguation_accuracies, datasets_with_predicted_labels


def evaluate_model_dir_path(models_dir_path, model_dir_path, model_epoch_dir_path):

    model, opts, parameters = initialize_model_with_pretrained_parameters(model_dir_path,
                                                                          model_epoch_dir_path,
                                                                          models_dir_path)

    # Prepare the data
    dev_data, dico_words_train, \
    id_to_tag, tag_scheme, test_data, \
    train_data, train_stats, word_to_id, \
    yuret_test_data, yuret_train_data = prepare_datasets(model,
                                                         opts,
                                                         parameters,
                                                         for_training=False)

    datasets_to_be_tested = [("dev", dev_data),
                             ("test", test_data)]

    f_scores, morph_accuracies, _ = predict_tags_given_model_and_input(datasets_to_be_tested,
                                                                       model,
                                                                       return_result=False)

    print(f_scores)
    print(morph_accuracies)


def predict_sentences_given_model(sentences_string, model):
    """

    :type sentences_string: string
    :type model: MainTaggerModel
    :param model:
        Mappings must be loaded.
    """

    from utils import tokenize_sentences_string
    from utils.loader import load_sentences, prepare_dataset

    tokenized_sentences = tokenize_sentences_string(sentences_string)

    # print tokenized_sentences

    from utils.morph_analyzer_caller import get_morph_analyzes, create_single_word_single_line_format

    # "\n".join([" ".join(x) for x in tokenized_sentences])
    dataset_file_string = ""
    morph_analyzer_output_for_all_sentences = ""
    for tokenized_sentence in tokenized_sentences:
        morph_analyzer_output_for_a_single_sentence = get_morph_analyzes(" ".join(tokenized_sentence))
        morph_analyzer_output_for_all_sentences += morph_analyzer_output_for_a_single_sentence + "\n"
        # print string_output
        dataset_file_string += create_single_word_single_line_format(morph_analyzer_output_for_a_single_sentence,
                                                                       conll=True,
                                                                       for_prediction=True)

    dataset_file_string = dataset_file_string.decode('iso-8859-9')
    # import sys
    # sys.exit(1)

    # print sentences_data_string.split("\n")
    # We now have the input sentences in our native format
    train_sentences, _, _ = load_sentences(dataset_file_string.split("\n"),
                                           model.parameters["zeros"])

    char_to_id, id_to_char, id_to_morpho_tag, id_to_tag, id_to_word, morpho_tag_to_id, tag_to_id, word_to_id = \
        extract_mapping_dictionaries_from_model(model)

    _, _, _, sentences_data = prepare_dataset(
        train_sentences,
        word_to_id, char_to_id, tag_to_id, morpho_tag_to_id,
        model.parameters['lower'],
        model.parameters['mt_d'], model.parameters['mt_t'], model.parameters['mt_ci'],
        morpho_tag_separator=("+" if model.parameters['lang_name'] == "turkish" else "|")
    )

    f_scores, morph_accuracies, labeled_sentences = \
        predict_tags_given_model_and_input([('tagger_output', sentences_data)],
                                           model,
                                           return_result=True)

    print(labeled_sentences)
    return labeled_sentences, dataset_file_string


def predict_tags_given_model_and_input(datasets_to_be_tested,
                                       model,
                                       return_result=False):

    f_scores, morph_accuracies, labeled_sentences = eval_with_specific_model(model,
                                                                             -1,
                                                                             datasets_to_be_tested,
                                                                             return_result)
    return f_scores, morph_accuracies, labeled_sentences


def initialize_model_with_pretrained_parameters(model_dir_path, model_epoch_dir_path, models_dir_path):
    import os
    from utils import read_parameters_from_file
    parameters, opts = read_parameters_from_file(os.path.join(model_dir_path, "parameters.pkl"),
                                                 os.path.join(model_dir_path, "opts.pkl"))
    model = MainTaggerModel(models_path=models_dir_path,
                            model_path=model_dir_path,
                            model_epoch_dir_path=model_epoch_dir_path)
    # Build the model
    model.build(training=False, **parameters)
    model.reload(model_epoch_dir_path)
    return model, opts, parameters


def evaluate(sys_argv):

    from utils import read_args

    opts = read_args(args_as_a_list=sys_argv[1:])

    from utils.train import models_path

    evaluate_model_dir_path(
        models_dir_path=models_path,
        model_dir_path=opts.model_path,
        model_epoch_dir_path=opts.model_epoch_path
    )


def predict_from_stdin(sys_argv):

    from utils import read_args

    opts = read_args(args_as_a_list=sys_argv[1:])

    from utils.train import models_path

    model, opts, parameters = initialize_model_with_pretrained_parameters(opts.model_path,
                                                                          opts.model_epoch_path,
                                                                          models_path)

    line = sys.stdin.readline()
    while line:
        # "ali ata bak\ndeneme deneme"
        predict_sentences_given_model(line.decode("utf8"), model)
        line = sys.stdin.readline()