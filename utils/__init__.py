
import codecs
import pickle
import optparse
import os
import re
import sys
from collections import OrderedDict

import numpy as np

models_path = "./models"
eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
# TODO: Move this to a better configurational structure
eval_logs_dir = os.path.join(eval_temp, "eval_logs")

if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)
if not os.path.exists(eval_logs_dir):
    os.makedirs(eval_logs_dir)

eval_script = os.path.join(eval_path, "conlleval-runner.sh")


class RegexpTokenizer():

    pattern = r"\w+[']\w+|\w+|\$[\d\.]+|\S+"
    flags = re.UNICODE | re.MULTILINE | re.DOTALL
    # flags = None

    regexp = None

    def __init__(self):
        self.regexp = re.compile(self.pattern, self.flags)

    def tokenize(self, sentence):
        return self.regexp.findall(sentence)


tokenizer = RegexpTokenizer()


def tokenize_sentences_string(sentences_string):
    """

    :type sentences_string: str
    """

    tokenized_sentences = []

    sentence_lines = sentences_string.split("\n")
    for sentences_string_line in sentence_lines:
        tokenized_sentences.append(tokenizer.tokenize(sentences_string_line))

    return tokenized_sentences


def lock_file(f):
    import fcntl, errno, time
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except IOError as e:
            # raise on unrelated IOErrors
            if e.errno != errno.EAGAIN:
                raise
            else:
                time.sleep(0.1)
    return True


def unlock_file(f):
    import fcntl
    fcntl.flock(f, fcntl.LOCK_UN)


def create_a_model_subpath(models_path):
    current_model_paths = read_model_paths_database(models_path)
    if len(current_model_paths) > 0:
        last_model_path_id_part = int(current_model_paths[-1][0].split("-")[1])
    else:
        last_model_path_id_part = -1

    return os.path.join(models_path, "model-%08d" % (last_model_path_id_part+1)), (last_model_path_id_part+1)


def add_a_model_path_to_the_model_paths_database(models_path, model_subpath, model_params_string):
    f = codecs.open(os.path.join(models_path, "model_paths_database.dat"), "a+")
    lock_file(f)
    f.write("%s %s\n" % (model_subpath, model_params_string))
    unlock_file(f)
    f.close()

def read_model_paths_database(models_path):
    try:
        f = codecs.open(os.path.join(models_path, "model_paths_database.dat"), "r")
        lock_file(f)
        lines = f.readlines()
        sorted_model_subpaths = sorted([line.strip().split() for line in lines if len(line.strip()) > 0], key=lambda x: x[0])
        # current_model_paths = {model_path: model_params for model_path, model_params in sorted_model_paths}
        unlock_file(f)
        f.close()
    except IOError as e:
        return []
    return sorted_model_subpaths

def get_model_subpath(parameters):
    model_parameters_string = get_name(parameters)
    sorted_model_subpaths = read_model_paths_database("models")
    for cur_model_subpath, cur_model_parameters_string in sorted_model_subpaths[::-1]:
        if cur_model_parameters_string == model_parameters_string:
            return cur_model_subpath


def get_name(parameters):
    """
    Generate a model name from its parameters.
    """
    l = []
    for k, v in list(parameters.items()):
        if (type(v) is str or type(v) is str) and "/" in v:
            l.append((k, v[::-1][:v[::-1].index('/')][::-1]))
        else:
            l.append((k, v))
    name = ",".join(["%s=%s" % (k, str(v).replace(',', '')) for k, v in l])
    return "".join(i for i in name if i not in "\/:*?<>|")


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(list(dico.items()), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in list(id_to_item.items())}
    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words


def pad_word_chars(words):
    """
    Pad the characters of the words in a sentence.
    Input:
        - list of lists of ints (list of words, a word being a list of char indexes)
    Output:
        - padded list of lists of ints
        - padded list of lists of ints (where chars are reversed)
        - list of ints corresponding to the index of the last character of each word
    """
    max_length = max([len(word) for word in words])
    char_for = []
    char_rev = []
    char_pos = []
    for word in words:
        padding = [0] * (max_length - len(word))
        char_for.append(word + padding)
        char_rev.append(word[::-1] + padding)
        char_pos.append(len(word) - 1)
    return char_for, char_rev, char_pos

def create_input(data, parameters, add_label, singletons=None):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    words = data['words']
    chars = data['chars']
    if singletons is not None:
        words = insert_singletons(words, singletons)
    if parameters['cap_dim']:
        caps = data['caps']
    char_for, char_rev, char_pos = pad_word_chars(chars)
    input = []

    if parameters['word_dim']:
        input.append(words)

    if parameters['char_dim']:
        input.append(char_for)
        if parameters['ch_b']:
            input.append(char_rev)
        input.append(char_pos)

    if parameters['cap_dim']:
        input.append(caps)
    # print input
    if add_label:
        input.append(data['tags'])
    # print input
    return input


def read_args(evaluation=False, args_as_a_list=sys.argv[1:], for_xnlp=False):
    optparser = optparse.OptionParser()

    if for_xnlp:
        optparser.add_option(
            "-r", "--reload", default="0",
            type='int', help="Reload the last saved model"
        )
        optparser.add_option(
            "--model_path", default="",
            type='str', help="Model path must be given when a reload is requested"
        )
        optparser.add_option(
            "--model_epoch_path", default="",
            type='str', help="Model epoch path must be given when a reload is requested"
        )
    else:
        for label in ["ner", "md"]:
            optparser.add_option(
                "--{label}_train_file".format(label=label), default="",
                help="Train set location"
            )
            optparser.add_option(
                "--{label}_dev_file".format(label=label), default="",
                help="Dev set location"
            )
            optparser.add_option(
                "--{label}_test_file".format(label=label), default="",
                help="Test set location"
            )

        optparser.add_option(
            "--lang_name", default="turkish",
            help="langugage name"
        )

        optparser.add_option(
            "--alt_dataset_group", default="none",
            help="alternative dataset group selector"
        )

        optparser.add_option(
            "--use_golden_morpho_analysis_in_word_representation", default=False, action="store_true",
            help="use golden morpho analysis when representing words"
        )
        optparser.add_option(
            "-s", "--tag_scheme", default="iobes",
            help="Tagging scheme (IOB or IOBES)"
        )
        optparser.add_option(
            "-l", "--lower", default="0",
            type='int', help="Lowercase words (this will not affect character inputs)"
        )
        optparser.add_option(
            "-z", "--zeros", default="0",
            type='int', help="Replace digits with 0"
        )
        optparser.add_option(
            "-c", "--char_dim", default="25",
            type='int', help="Char embedding dimension"
        )
        optparser.add_option(
            "-C", "--char_lstm_dim", default="25",
            type='int', help="Char LSTM hidden layer size"
        )
        optparser.add_option(
            "-b", "--char_bidirect", default="1",
            type='int', help="Use a bidirectional LSTM for chars"
        )
        # morpho_tag section
        optparser.add_option(
            "--morpho_tag_dim", default="100",
            type='int', help="Morpho tag embedding dimension"
        )
        optparser.add_option(
            "--morpho_tag_lstm_dim", default="100",
            type='int', help="Morpho tag LSTM hidden layer size"
        )
        optparser.add_option(
            "--morpho_tag_bidirect", default="1",
            type='int', help="Use a bidirectional LSTM for morpho tags"
        )
        optparser.add_option(
            "--morpho_tag_type", default="char",
            help="Mode of morphological tag extraction"
        )
        optparser.add_option(
            "--morpho-tag-column-index", default="1",
            type='int', help="the index of the column which contains the morphological tags in the conll format"
        )
        optparser.add_option(
            "--integration_mode", default="0",
            type='int', help="integration mode"
        )
        optparser.add_option(
            "--active_models", default="0",
            type='int', help="active models: 0: NER, 1: MD, 2: JOINT"
        )
        optparser.add_option(
            "--multilayer", default="0",
            type='int', help="use a multilayered sentence level Bi-LSTM"
        )
        optparser.add_option(
            "--shortcut_connections", default="0",
            type='int', help="use shortcut connections in the multilayered scheme"
        )
        optparser.add_option(
            "--tying_method", default="",
            help="tying method"
        )
        optparser.add_option(
            "-w", "--word_dim", default="100",
            type='int', help="Token embedding dimension"
        )
        optparser.add_option(
            "-W", "--word_lstm_dim", default="100",
            type='int', help="Token LSTM hidden layer size"
        )
        optparser.add_option(
            "-B", "--word_bidirect", default="1",
            type='int', help="Use a bidirectional LSTM for words"
        )
        optparser.add_option(
            "-p", "--pre_emb", default="",
            help="Location of pretrained embeddings"
        )
        optparser.add_option(
            "-A", "--all_emb", default="0",
            type='int', help="Load all embeddings"
        )
        optparser.add_option(
            "-a", "--cap_dim", default="0",
            type='int', help="Capitalization feature dimension (0 to disable)"
        )
        optparser.add_option(
            "-f", "--crf", default="1",
            type='int', help="Use CRF (0 to disable)"
        )
        optparser.add_option(
            "-D", "--dropout", default="0.5",
            type='float', help="Droupout on the input (0 = no dropout)"
        )
        optparser.add_option(
            "-L", "--lr_method", default="adam-alpha_float@0.005",
            help="Learning method (SGD, Adadelta, Adam..)"
        )
        optparser.add_option(
            "--disable_sparse_updates", default=True, action="store_false",
            dest="sparse_updates_enabled",
            help="Sparse updates enabled"
        )
        optparser.add_option(
            "-r", "--reload", default="0",
            type='int', help="Reload the last saved model"
        )
        optparser.add_option(
            "--model_path", default="",
            type='str', help="Model path must be given when a reload is requested"
        )
        optparser.add_option(
            "--model_epoch_path", default="",
            type='str', help="Model epoch path must be given when a reload is requested"
        )
        optparser.add_option(
            "--skip-testing", default="0",
            type='int',
            help="Skip the evaluation on test set (because dev and test sets are the same and thus testing is irrelevant)"
        )
        optparser.add_option(
            "--predict-and-exit-filename", default="",
            help="Used with '--reload 1', the loaded model is used for predicting on the test set and the results are written to the filename"
        )
        optparser.add_option(
            "--overwrite-mappings", default="0",
            type='int', help="Explicitly state to overwrite mappings"
        )
        optparser.add_option(
            "--starting-epoch-no", default="1",
            type='int', help="Starting epoch no for resuming training"
        )
        optparser.add_option(
            "--maximum-epochs", default="100",
            type='int', help="Maximum number of epochs"
        )
        optparser.add_option(
            "--batch-size", default="5",
            type='int', help="Number of samples in one epoch"
        )
        optparser.add_option(
            "--file_format", default="conll", choices=["conll", "conllu"],
            help="File format of the data files"
        )
        optparser.add_option(
            "--debug", default="0",
            type='int', help="whether to print lots of debugging info."
        )
        optparser.add_option(
            "--dynet-gpu", default="1",
            type='int', help="Use gpu or not"
        )
        optparser.add_option(
            "--port", default="8888",
            type='int', help="Webapp port to serve on localhost"
        )
        if evaluation:
            optparser.add_option(
                "--run-for-all-checkpoints", default="0",
                type='int', help="run evaluation for all checkpoints"
            )
    opts = optparser.parse_args(args_as_a_list)[0]
    return opts


def form_parameters_dict(opts):
    parameters = OrderedDict()
    parameters['t_s'] = opts.tag_scheme
    parameters['lower'] = opts.lower == 1
    parameters['zeros'] = opts.zeros == 1
    parameters['char_dim'] = opts.char_dim
    parameters['char_lstm_dim'] = opts.char_lstm_dim
    parameters['ch_b'] = opts.char_bidirect == 1

    # morpho_tag section
    parameters['mt_d'] = opts.morpho_tag_dim
    parameters['mt_t'] = opts.morpho_tag_type
    parameters['mt_ci'] = opts.morpho_tag_column_index
    parameters['integration_mode'] = opts.integration_mode
    parameters['active_models'] = opts.active_models

    parameters['multilayer'] = opts.multilayer
    parameters['shortcut_connections'] = opts.shortcut_connections

    parameters['tying_method'] = opts.tying_method

    parameters['use_golden_morpho_analysis_in_word_representation'] = opts.use_golden_morpho_analysis_in_word_representation

    parameters['word_dim'] = opts.word_dim
    parameters['word_lstm_dim'] = opts.word_lstm_dim
    parameters['w_b'] = opts.word_bidirect == 1
    parameters['pre_emb'] = opts.pre_emb
    parameters['all_emb'] = opts.all_emb == 1
    parameters['cap_dim'] = opts.cap_dim
    parameters['crf'] = opts.crf == 1
    parameters['dropout'] = opts.dropout
    parameters['lr_method'] = opts.lr_method
    parameters['sparse_updates_enabled'] = opts.sparse_updates_enabled

    parameters['batch_size'] = opts.batch_size

    parameters['file_format'] = opts.file_format
    parameters['lang_name'] = opts.lang_name

    parameters['debug'] = 1 if opts.debug == 1 else 0

    return parameters


def read_parameters_from_file(filepath, opts_filepath):

    with open(filepath, "rb") as f:
        parameters = pickle.load(f)

    with open(opts_filepath, "rb") as f:
        opts = pickle.load(f)

    return parameters, opts


def read_parameters_from_sys_argv(sys_argv):
    opts = read_args(args_as_a_list=sys_argv[1:])
    # Parse parameters
    parameters = form_parameters_dict(opts)

    # Check parameters validity
    parameters = check_parameter_validity(parameters)

    return opts, parameters


def check_parameter_validity(parameters):

    assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
    assert 0. <= parameters['dropout'] < 1.0
    assert parameters['t_s'] in ['iob', 'iobes']
    assert not parameters['all_emb'] or parameters['pre_emb']
    assert not parameters['pre_emb'] or parameters['word_dim'] > 0
    assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])

    return parameters