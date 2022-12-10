# coding=utf-8
from functools import partial
import itertools
import json
import os
import re
import codecs

from utils import create_dico, create_mapping, zero_digits
from utils import iob2, iob_iobes

from toolkit.joint_ner_and_md_model import MainTaggerModel

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sentences(input_file_path_or_list, zeros, file_format="conll"):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """

    assert file_format in ["conll", "conllu"]

    sentences = []
    sentence = []
    max_sentence_length = 0
    max_word_length = 0

    if isinstance(input_file_path_or_list, str):
        input_f = codecs.open(input_file_path_or_list, 'r', 'utf8')
    else:
        input_f = input_file_path_or_list

    if file_format == "conllu":
        sep = '\t'
    elif file_format == "conll":
        sep = ' '

    for line in input_f:
        if file_format == "conllu" and line.startswith("#"):
            continue
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    # print sentence
                    # sys.exit()
                    sentences.append(sentence)
                    if len(sentence) > max_sentence_length:
                        max_sentence_length = len(sentence)
                sentence = []
        else:
            tokens = line.split(sep)
            if file_format == "conll":
                assert len(tokens) >= 2
            elif file_format == "conllu":
                assert len(tokens) == 10, line + " " + " ".join(tokens) + " CONLL-U format requires exactly 10 columns"
                if "-" in tokens[0]: # skip if the first column contains '-' as this indicates that this line is irrelavant for us.
                    continue
            sentence.append(tokens)
            if len(tokens[0]) > max_word_length:
                max_word_length = len(tokens[0])
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
            if len(sentence) > max_sentence_length:
                max_sentence_length = len(sentence)
    return sentences, max_sentence_length, max_word_length


def update_tag_scheme(sentences, tag_scheme, file_format="conll"):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = []
        if file_format == "conll":
            tags = [w[-1] for w in s]
        elif file_format == "conllu":
            if contains_golden_label(s[0], "NER_TAG"):
                tags = [extract_correct_ner_tag_from_conllu(w) for w in s]
            else:
                continue
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            print(s_str.encode("utf8"))
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                if file_format == "conll":
                    word[-1] = new_tag
                elif file_format == "conllu":
                    field_contents_dict = load_MISC_column_contents(word[9])
                    field_contents_dict["NER_TAG"] = new_tag
                    word[9] = compile_MISC_column_contents(field_contents_dict)
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                if file_format == "conll":
                    word[-1] = new_tag
                elif file_format == "conllu":
                    field_contents_dict = load_MISC_column_contents(word[9])
                    field_contents_dict["NER_TAG"] = new_tag
                    word[9] = compile_MISC_column_contents(field_contents_dict)
        else:
            raise Exception('Unknown tagging scheme!')


def word_mapping(sentences, lower, file_format="conll"):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    # words = [[(" ".join(x[0:2])).lower() if lower else " ".join(x[0:2]) for x in s] for s in sentences]
    surface_form_index = 0
    if file_format == "conll":
        surface_form_index = 0
    elif file_format == "conllu":
        surface_form_index = 1
    words = [[x[surface_form_index].lower() if lower else x[surface_form_index] for x in s] for s in sentences]
    # TODO: only roots version, but this effectively damages char embeddings.
    # words = [[x[1].split("+")[0].lower() if lower else x[1].split("+")[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    ))
    return dico, word_to_id, id_to_word


def char_mapping(sentences, file_format="conll"):
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    surface_form_index = 0
    chars = []
    if file_format == "conll":
        surface_form_index = 0
        chars = ["".join([w[surface_form_index] + "".join(w[2:-1]) for w in s]) for s in sentences]
    elif file_format == "conllu":
        surface_form_index = 1
        chars = ["".join([w[surface_form_index] for w in s]) for s in sentences]
    chars.append("+")
    chars.append("*")
    dico = create_dico(chars)
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique characters" % len(dico))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences, file_format="conll"):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = []
    if file_format == "conll":
        tags = [[word[-1] for word in s] for s in sentences]
    elif file_format == "conllu":
        tags = [[extract_correct_ner_tag_from_conllu(word) for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def load_MISC_column_contents(column):
    # fields_dict = {}
    # fields = column.split("|")
    # for field in fields:
    #     tokens = field.split("=")
    #     if len(tokens) == 2:
    #         field_name = tokens[0]
    #         field_content = [item for item in tokens[1].split("!")]
    #         fields_dict[field_name] = field_content
    try:
        fields_dict = json.loads(column)
    except json.decoder.JSONDecodeError as e:
        return None
    return fields_dict

def compile_MISC_column_contents(field_contents_dict):
    # field_contents_str = ""
    # for field_label in field_contents_dict.keys():
    #     field_contents_str += field_label + "=" + \
    #                           "!".join(field_contents_dict[field_label]) \
    #                           + "|"
    # field_contents_str = field_contents_str[:-1]
    field_contents_str = json.dumps(field_contents_dict, separators=(',', ':'))
    return field_contents_str


def morpho_tag_mapping(sentences, morpho_tag_type='wo_root', morpho_tag_column_index=1,
                       joint_learning=False,
                       file_format="conll",
                       morpho_tag_separator="+"):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    if file_format == "conll":

        if morpho_tag_type == 'char':
            morpho_tags = ["".join([w[morpho_tag_column_index] for w in s]) for s in sentences]
            morpho_tags += [ww for ww in w[2:-1] for w in s for s in sentences]
        else:
            morpho_tags = extract_morpho_tags_ordered(morpho_tag_type,
                                                      sentences, morpho_tag_column_index,
                                                      joint_learning=joint_learning,
                                                      file_format=file_format,
                                                      morpho_tag_separator=morpho_tag_separator,
                                                      use_all_analyses=True)

    elif file_format == "conllu":

        if morpho_tag_type == 'char':
            # extract CORRECT_ANALYSIS and ALL_ANALYSES fields from column 10

            morpho_tags = ["".join([extract_correct_analysis_from_conllu(w) for w in s]) for s in sentences]
            _tmp_morpho_tags = [[load_MISC_column_contents(w[9]) for w in s] for s in sentences]
            morpho_tags += ["".join(["".join([analysis for analysis in misc_column_contents["ALL_ANALYSES"]])
                                     for misc_column_contents in s if "ALL_ANALYSES" in misc_column_contents])
                            for s in _tmp_morpho_tags]
        else:
            morpho_tags = extract_morpho_tags_ordered(morpho_tag_type,
                                                      sentences, morpho_tag_column_index,
                                                      joint_learning=joint_learning,
                                                      file_format=file_format,
                                                      morpho_tag_separator=morpho_tag_separator,
                                                      use_all_analyses=True)


        ## TODO: xxx

    # print morpho_tags
    #morpho_tags = [[word[1].split("+") for word in s] for s in sentences]
    # print morpho_tags
    morpho_tags.append(["*UNKNOWN*"])
    dico = create_dico(morpho_tags)
    # print dico
    morpho_tag_to_id, id_to_morpho_tag = create_mapping(dico)
    print(morpho_tag_to_id)
    print("Found %i unique morpho tags" % len(dico))
    return dico, morpho_tag_to_id, id_to_morpho_tag


def extract_morpho_tags_ordered(morpho_tag_type,
                                sentences, morpho_tag_column_index,
                                joint_learning=False,
                                file_format="conll",
                                morpho_tag_separator="+",
                                use_all_analyses=False):
    morpho_tags = []
    for sentence in sentences:
        # print s
        # sys.exit(1)
        morpho_tags += extract_morpho_tags_from_one_sentence_ordered(morpho_tag_type, [], sentence,
                                                                     morpho_tag_column_index,
                                                                     file_format=file_format,
                                                                     morpho_tag_separator=morpho_tag_separator,
                                                                     use_all_analyses=use_all_analyses)
    return morpho_tags


def extract_morpho_tags_from_one_sentence_ordered(morpho_tag_type, morpho_tags, sentence, morpho_tag_column_index,
                                                  file_format="conll",
                                                  morpho_tag_separator="+",
                                                  use_all_analyses=False):
    assert morpho_tag_column_index in [1, 2], "We expect to 1 or 2"

    def fix_BLANK(x):
        if x == '':
            return "*BLANK*"
        else:
            return x
    print(sentence)
    print(morpho_tag_type)
    print(morpho_tags)
    print(morpho_tag_column_index)
    print(file_format)
    print(morpho_tag_separator)
    print(use_all_analyses)
    for word in sentence:
        if morpho_tag_type.startswith('wo_root'):
            if morpho_tag_type == 'wo_root_after_DB' and morpho_tag_column_index == 1: # this is only applicable to Turkish dataset
                tmp = []
                if file_format == "conll":
                    tmp_morpho_tag = word[1]
                elif file_format == "conllu":
                    tmp_morpho_tag = extract_correct_analysis_from_conllu(word)
                for tag in tmp_morpho_tag.split(morpho_tag_separator)[1:][::-1]:
                    if tag.endswith("^DB"):
                        tmp += [tag]
                        break
                    else:
                        tmp += [tag]
                morpho_tags += [tmp]
            else:
                if morpho_tag_column_index == 2: # this means we're reading Czech dataset (it's faulty in a sense)
                    morpho_tags += [word[morpho_tag_column_index].split("")]
                else:
                    if file_format == "conll":
                        tmp_morpho_tag = word[morpho_tag_column_index]
                    elif file_format == "conllu":
                        if use_all_analyses:
                            for tmp_morpho_tag in extract_all_analyses_from_conllu(word):
                                if len(tmp_morpho_tag.split(morpho_tag_separator)) == 1:
                                    tmp_morpho_tag = morpho_tag_separator.join([tmp_morpho_tag, "*UNKNOWN*"])
                                morpho_tags += [list(map(fix_BLANK, [tmp_morpho_tag.split(morpho_tag_separator)[1].split("~")[
                                                     -1]] + tmp_morpho_tag.split(morpho_tag_separator)[2:]))]
                        tmp_morpho_tag = extract_correct_analysis_from_conllu(word)
                        if len(tmp_morpho_tag.split(morpho_tag_separator)) == 1:
                            tmp_morpho_tag = morpho_tag_separator.join([tmp_morpho_tag, "*UNKNOWN*"])
                    print(word)
                    print(tmp_morpho_tag)
                    if tmp_morpho_tag == "_":
                        morpho_tags = []
                    else:
                        morpho_tags += [list(map(fix_BLANK, [tmp_morpho_tag.split(morpho_tag_separator)[1].split("~")[-1]] + tmp_morpho_tag.split(morpho_tag_separator)[2:]))]
        elif morpho_tag_type.startswith('with_root'):
            print("word: ", word)
            if morpho_tag_column_index == 1:
                if file_format == "conll":
                    tmp_morpho_tag = word[morpho_tag_column_index]
                elif file_format == "conllu":
                    tmp_morpho_tag = extract_correct_analysis_from_conllu(word)
                root = [tmp_morpho_tag.split(morpho_tag_separator)[0]]
            else:
                root = [word[1]] # In Czech dataset, the lemma is given in the first column
            tmp = []
            tmp += root
            print("tmp: ", tmp)
            if morpho_tag_type == 'with_root_after_DB' and morpho_tag_column_index == 1:
                if file_format == "conll":
                    tmp_morpho_tag = word[1]
                elif file_format == "conllu":
                    tmp_morpho_tag = extract_correct_analysis_from_conllu(word)
                for tag in tmp_morpho_tag.split(morpho_tag_separator)[1:][::-1]:
                    if tag.endswith("^DB"):
                        tmp += [tag]
                        break
                    else:
                        tmp += [tag]
                morpho_tags += [tmp]
            else:
                if morpho_tag_column_index == 2:
                    morpho_tags += [tmp + word[morpho_tag_column_index].split("")]
                else: # only 1 is possible
                    if file_format == "conll":
                        tmp_morpho_tag = word[1]
                    elif file_format == "conllu":
                        tmp_morpho_tag = extract_correct_analysis_from_conllu(word)
                    print("tmp_morpho_tag: ", tmp_morpho_tag)
                    morpho_tags += [tmp_morpho_tag.split(morpho_tag_separator)] # I removed the 'tmp +' because it just repeated the first element which is root
    return morpho_tags


def contains_golden_label(word, type):
    if len(word) == 10:
        misc_dict = load_MISC_column_contents(word[9])
        if misc_dict:
            return type in list(misc_dict.keys())
        else:
            return False
    else:
        return False


def extract_specific_single_field_content_from_conllu(word, field_name):
    misc_dict = load_MISC_column_contents(word[9])
    if field_name in misc_dict:
        return misc_dict[field_name]
    else:
        return None


def extract_correct_analysis_from_conllu(word):
    misc_dict = load_MISC_column_contents(word[9])
    if "CORRECT_ANALYSIS" in misc_dict:
        return misc_dict["CORRECT_ANALYSIS"]
    else:
        return "_"
    # return extract_specific_single_field_content_from_conllu(word, "CORRECT_ANALYSIS")


def extract_correct_ner_tag_from_conllu(word):
    return extract_specific_single_field_content_from_conllu(word, "NER_TAG")


def extract_all_analyses_from_conllu(word):
    misc_dict = load_MISC_column_contents(word[9])
    if "ALL_ANALYSES" in misc_dict:
        return misc_dict["ALL_ANALYSES"]
    else:
        return []


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """

    def cap_characterization(input_s):
        if input_s.lower() == input_s:
            return 0
        elif input_s.upper() == input_s:
            return 1
        elif input_s[0].upper() == input_s[0]:
            return 2
        elif sum([x == y for (x, y) in zip(input_s.upper(), input_s)]) > 0:
            return 3

    if is_number(s):
        return 0
    elif sum([(str(digit) in s) for digit in range(0, 10)]) > 0:
        if "'" in s:
            return 1 + cap_characterization(s)
        else:
            return 1 + 4 + cap_characterization(s)
    else:
        if "'" in s:
            return 1 + 8 + cap_characterization(s)
        else:
            return 1 + 12 + cap_characterization(s)


def prepare_sentence(str_words, word_to_id, char_to_id, lower=False):
    """
    Prepare a sentence for evaluation.
    """
    def f(x): return x.lower() if lower else x
    words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
             for w in str_words]
    chars = [[char_to_id[c] for c in w if c in char_to_id]
             for w in str_words]
    caps = [cap_feature(w) for w in str_words]
    return {
        'str_words': str_words,
        'words': words,
        'chars': chars,
        'caps': caps
    }


def turkish_lower(s):
    return s.replace("IİŞÜĞÖÇ", "ıişüğöç")


def prepare_dataset(sentences,
                    word_to_id, char_to_id, tag_to_id, morpho_tag_to_id,
                    lower=False,
                    morpho_tag_dimension=0,
                    morpho_tag_type='wo_root',
                    morpho_tag_column_index=1,
                    file_format="conll",
                    morpho_tag_separator="+",
                    for_prediction=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """

    def lower_or_not(x): return x.lower() if lower else x
    data = []

    for sentence in sentences:
        # surface form related
        surface_form_index = 0
        if file_format == "conll":
            surface_form_index = 0
        elif file_format == "conllu":
            surface_form_index = 1

        punctuation_marks = "` = - , ; : / . \" ( ) +".split(" ")
        for w in sentence:
            if "Punc" in w[2] and w[surface_form_index] not in punctuation_marks:
                w[morpho_tag_column_index] = ".+Punc"
                w[surface_form_index] = "."

        surface_forms = [w[surface_form_index] for w in sentence]
        #####

        # word indexing related
        words = [word_to_id[lower_or_not(surface_form) if lower_or_not(surface_form) in word_to_id else '<UNK>']
                 for surface_form in surface_forms]
        #####
        # char indexing related
        # Skip characters that are not in the training set
        chars = [[char_to_id[c] for c in surface_form if c in char_to_id]
                 for surface_form in surface_forms]
        #####
        # capitalization related
        caps = [cap_feature(surface_form) for surface_form in surface_forms]
        #####

        # NER label indexing related
        ner_labels = []
        if file_format == "conll":
            ner_labels = [tag_to_id[w[-1]] for w in sentence]
        elif file_format == "conllu" and contains_golden_label(sentence[0], "NER_TAG"):
            ner_labels = [tag_to_id[extract_correct_ner_tag_from_conllu(w)] for w in sentence]
        #####

        # if contains_golden_label(sentence[0], "CORRECT_ANALYSIS"):

        # MD tag indexing related
        if morpho_tag_dimension > 0:
            if file_format == "conll":
                if morpho_tag_type == 'char':
                    str_morpho_tags = [w[morpho_tag_column_index] for w in sentence]
                    morpho_tags = [[morpho_tag_to_id[c] for c in str_morpho_tag if c in morpho_tag_to_id]
                         for str_morpho_tag in str_morpho_tags]
                else:
                    morpho_tags_in_the_sentence = \
                        extract_morpho_tags_from_one_sentence_ordered(morpho_tag_type, [], sentence,
                                                                      morpho_tag_column_index,
                                                                      file_format=file_format,
                                                                      morpho_tag_separator=morpho_tag_separator)

                    morpho_tags = [[morpho_tag_to_id[morpho_tag] for morpho_tag in ww if morpho_tag in morpho_tag_to_id]
                                   for ww in morpho_tags_in_the_sentence]
            elif file_format  == "conllu":
                if contains_golden_label(sentence[0], "CORRECT_ANALYSIS"):
                    if morpho_tag_type == 'char':
                        str_morpho_tags = [extract_correct_analysis_from_conllu(w) for w in sentence]
                        morpho_tags = [[morpho_tag_to_id[c] for c in str_morpho_tag if c in morpho_tag_to_id]
                                       for str_morpho_tag in str_morpho_tags]
                    else:
                        morpho_tags_in_the_sentence = \
                            extract_morpho_tags_from_one_sentence_ordered(morpho_tag_type, [], sentence,
                                                                          morpho_tag_column_index,
                                                                          file_format=file_format,
                                                                          morpho_tag_separator=morpho_tag_separator)

                        morpho_tags = [[morpho_tag_to_id[morpho_tag] for morpho_tag in ww if morpho_tag in morpho_tag_to_id]
                                       for ww in morpho_tags_in_the_sentence]
                else:
                    morpho_tags = []
        #####

        def f_morpho_tag_to_id(m):
            if m in morpho_tag_to_id:
                return morpho_tag_to_id[m]
            else:
                return morpho_tag_to_id['*UNKNOWN*']

        # All candidate morphological analyses

        def replace_if_None(x, replacement):
            if x is None:
                return replacement
            else:
                x

        all_analyses = []
        if file_format == "conll":
            correct_analyses = [w[morpho_tag_column_index] for w in sentence]
            all_analyses = [w[2:-1] for w in sentence]
        elif file_format == "conllu":
            correct_analyses = [extract_correct_analysis_from_conllu(w) for w in sentence]
            for i in range(len(correct_analyses)):
                if correct_analyses[i] is None:
                    correct_analyses[i] = ""
            all_analyses = [extract_all_analyses_from_conllu(w) for w in sentence]
            for i in range(len(all_analyses)):
                if all_analyses[i] is None:
                    all_analyses[i] = []

        if len(all_analyses) > 0 and len(all_analyses[0]) == 0:
            print("ERROR IN ALL_ANALYSES")

        # for now we ignore different schemes we did in previous morph. tag parses.
        morph_analyses_tags = [] # list of list of lists
        for analyses_for_word in all_analyses:
            encoded_analyses_for_word = []
            for analysis in analyses_for_word:
                if morpho_tag_type == "char":
                    current_tag_sequence = list(morpho_tag_separator.join(analysis.split(morpho_tag_separator)[1:]))
                else:
                    if len(analysis.split(morpho_tag_separator)) == 1:
                        analysis = morpho_tag_separator.join([analysis, "*UNKNOWN*"])
                    current_tag_sequence = [analysis.split(morpho_tag_separator)[1].split("~")[-1]] + analysis.split(morpho_tag_separator)[2:]
                if current_tag_sequence:
                    encoded_analysis_for_word = [list(map(f_morpho_tag_to_id, current_tag_sequence))]
                else:
                    encoded_analysis_for_word = [[morpho_tag_to_id["*UNKNOWN*"]]]
                encoded_analyses_for_word += encoded_analysis_for_word
            morph_analyses_tags += [encoded_analyses_for_word]

        def f_char_to_id(c):
            if c in char_to_id:
                return char_to_id[c]
            else:
                return char_to_id['*']

        morph_analyses_roots = [[list(map(f_char_to_id, list(analysis.split(morpho_tag_separator)[0]))) \
                                     if list(analysis.split(morpho_tag_separator)[0]) else [char_to_id[morpho_tag_separator]]
                                for analysis in analyses] for analyses in all_analyses]

        # morph_analysis_from_NER_data = [w[morpho_tag_column_index] for w in s]
        # morph_analyzes_from_FST_unprocessed = [w[2:-1] for w in s]

        def remove_Prop_and_lower(s):
            return turkish_lower(s.replace("+Prop", ""))

        golden_analysis_indices = []
        if file_format == "conll" or (file_format == "conllu"):
            for w_idx in range(len(sentence)):
                if not(contains_golden_label(sentence[w_idx], "CORRECT_ANALYSIS") and contains_golden_label(sentence[w_idx], "ALL_ANALYSES")):
                    golden_analysis_idx = 0
                else:
                    found = False
                    try:
                        golden_analysis_idx = \
                            all_analyses[w_idx]\
                                .index(correct_analyses[w_idx])
                        found = True
                    except ValueError as e:
                        # step 1
                        pass
                    if not found:
                        try:
                            golden_analysis_idx = \
                                list(map(remove_Prop_and_lower, all_analyses[w_idx]))\
                                    .index(remove_Prop_and_lower(correct_analyses[w_idx]))
                            found = True
                        except ValueError as e:
                            pass
                    if not found:
                        if len(all_analyses[w_idx]) <= 1:
                            golden_analysis_idx = 0
                        else:
                            # WE expect that this never happens in gungor.ner.14.* files as they have been processed for unfound golden analyses
                            import random
                            golden_analysis_idx = random.randint(0, len(all_analyses[w_idx])-1)
                    if golden_analysis_idx >= len(all_analyses[w_idx]) or \
                        golden_analysis_idx < 0 or \
                        golden_analysis_idx >= len(morph_analyses_roots[w_idx]):
                        logging.error("BEEP at golden analysis idx")
                golden_analysis_indices.append(golden_analysis_idx)

        data_item = {
            'str_words': surface_forms,

            'word_ids': words,
            'char_for_ids': chars,
            'cap_ids': caps,

            'morpho_analyzes_tags': morph_analyses_tags,
            'morpho_analyzes_roots': morph_analyses_roots,

            'char_lengths': [len(char) for char in chars],
            'sentence_lengths': len(sentence),
            'max_word_length_in_this_sample': max([len(x) for x in chars])
        }

        if contains_golden_label(sentence[0], "NER_TAG"):
            data_item['tag_ids'] = ner_labels
        elif for_prediction:
            data_item['tag_ids'] = [tag_to_id['O'] for _ in range(len(words))]
        else:
            data_item['tag_ids'] = []

        # This is always added because they are not labels, they can be computed deterministically
        if morpho_tag_dimension > 0:
            data_item['morpho_tag_ids'] = morpho_tags
            if file_format == "conll" or (
                    file_format == "conllu" and contains_golden_label(sentence[0], "CORRECT_ANALYSIS")):
                data_item['golden_morph_analysis_indices'] = golden_analysis_indices

        if len(morph_analyses_tags) == 0:
            print("ERROR1")
        for morph_analyses_tags_for_word in morph_analyses_tags:
            if any([len(tag_sequence) == 0 for tag_sequence in morph_analyses_tags_for_word]):
                print("ERROR2")
        if len(morph_analyses_roots) == 0:
            print("ERROR3")
        for morph_analyses_roots_for_word in morph_analyses_roots:
            if any([len(root_sequence) == 0 for root_sequence in morph_analyses_roots_for_word]):
                print("ERROR4")

        data.append(data_item)

    logging.info("Sorting the dataset by sentence length..")
    data_sorted_by_sentence_length = sorted(data, key=lambda x: x['sentence_lengths'])
    stats = [[data_item['sentence_lengths'],
              data_item['max_word_length_in_this_sample'],
              data_item['char_lengths']] for data_item in data]
    n_unique_words = set()
    for data_item in data:
        for word_id in data_item['word_ids']:
            n_unique_words.add(word_id)
    n_unique_words = len(n_unique_words)

    n_buckets = min([9, len(sentences)])
    print("n_sentences: %d" % len(sentences))
    n_samples_to_be_bucketed = int(len(sentences)/n_buckets)

    print("n_samples_to_be_binned: %d" % n_samples_to_be_bucketed)

    buckets = []
    for bin_idx in range(n_buckets+1):
        logging.info("Forming bin %d.." % bin_idx)
        data_to_be_bucketed = data_sorted_by_sentence_length[n_samples_to_be_bucketed*(bin_idx):n_samples_to_be_bucketed*(bin_idx+1)]
        if len(data_to_be_bucketed) == 0:
            continue

        buckets.append(data_to_be_bucketed)

    return buckets, stats, n_unique_words, data


def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def _prepare_datasets(opts, parameters, for_training=True, do_xnlp=False, alt_dataset_group="none"):

    opts_dict = opts.__dict__

    # Data parameters
    lower = parameters['lower']
    zeros = parameters['zeros']
    tag_scheme = parameters['t_s']
    max_sentence_lengths = {}
    max_word_lengths = {}

    training_sets = {"ner": {}, "md": {}}

    if alt_dataset_group == "none":
        alt_dataset_group = ""
    else:
        alt_dataset_group = "." + alt_dataset_group

    # Load sentences
    if for_training or do_xnlp:
        for label in list(training_sets.keys()):
            _train_sentences, max_sentence_lengths['train'], max_word_lengths['train'] = \
                load_sentences(opts_dict[label+"_train_file"]+alt_dataset_group, zeros, parameters['file_format'])
            update_tag_scheme(_train_sentences, tag_scheme, file_format=parameters['file_format'])
            training_sets[label]['train'] = _train_sentences

    for label in list(training_sets.keys()):
        for purpose in ["dev", "test"]:
            if os.path.exists(opts_dict[label+"_"+purpose+"_file"]+alt_dataset_group):
                _dev_sentences, max_sentence_lengths[purpose], max_word_lengths[purpose] = \
                    load_sentences(opts_dict[label+"_"+purpose+"_file"]+alt_dataset_group, zeros, parameters['file_format'])
                update_tag_scheme(_dev_sentences, tag_scheme, file_format=parameters['file_format'])
                training_sets[label][purpose] = _dev_sentences

    return training_sets, max_sentence_lengths, max_word_lengths


def create_mappings(training_sets, parameters, file_format="conll",
                    morpho_tag_separator="+"):
    # Create a dictionary / mapping of words
    # If we use pretrained embeddings, we add them to the dictionary.
    if parameters['pre_emb']:
        dico_words_train = word_mapping(training_sets['ner']['train'] + training_sets['md']['train'],
                                        parameters['lower'],
                                        file_format=file_format)[0]
        dico_words, word_to_id, id_to_word = augment_with_pretrained(
            dico_words_train.copy(),
            parameters['pre_emb'],
            list(itertools.chain.from_iterable(
                [[w[0] for w in s] for s in
                 training_sets['ner']['dev'] + training_sets['md']['dev'] +
                 training_sets['ner']['test'] + training_sets['md']['test']])
            ) if not parameters['all_emb'] else None
        )
    else:
        dico_words, word_to_id, id_to_word = word_mapping(training_sets['ner']['train'] + training_sets['md']['train'],
                                                          parameters['lower'], file_format=file_format)
        dico_words_train = dico_words

    sentences_for_mapping = []
    for label in "ner md".split(" "):
        for purpose in "train dev test".split(" "):
            if label in training_sets and purpose in training_sets[label]:
                sentences_for_mapping += training_sets[label][purpose]

    # sentences_for_mapping = training_sets['ner']['train'] + training_sets['md']['train'] + \
    #                          training_sets['ner']['dev'] + training_sets['md']['dev'] + \
    #                          training_sets['ner']['test'] + training_sets['md']['test']
    # Create a dictionary and a mapping for words / POS tags / tags
    dico_chars, char_to_id, id_to_char = \
        char_mapping(sentences_for_mapping, file_format=file_format)
    dico_tags, tag_to_id, id_to_tag = \
        tag_mapping(training_sets["ner"]["train"]
                    + training_sets["ner"]["dev"]
                    + training_sets["ner"]["test"],
                    file_format=file_format)
    # if file_format is conllu, this works for datasets that contain CORRECT_ANALYSIS in 10th column
    if parameters['mt_d'] > 0:
        dico_morpho_tags, morpho_tag_to_id, id_to_morpho_tag = \
            morpho_tag_mapping(
                sentences_for_mapping,
                morpho_tag_type=parameters['mt_t'],
                morpho_tag_column_index=parameters['mt_ci'],
                joint_learning=True,
                file_format=file_format,
                morpho_tag_separator=morpho_tag_separator)
    else:
        id_to_morpho_tag = {}
        morpho_tag_to_id = {}

    return word_to_id, id_to_word,\
           char_to_id, id_to_char, \
           tag_to_id, id_to_tag, \
           morpho_tag_to_id, id_to_morpho_tag


def prepare_datasets(model, opts, parameters, for_training=True, do_xnlp=False):
    """

    :type model: MainTaggerModel
    :param model: description
    :param opts:
    :param parameters:
    :param for_training:
    :return:
    """

    ud_morpho_tag_separator = "|"

    if "alt_dataset_group" in opts.__dict__:
        alt_dataset_group = opts.alt_dataset_group
    else:
        alt_dataset_group = "none"

    training_sets, max_sentence_lengths, max_word_lengths = \
        _prepare_datasets(opts, parameters,
                          for_training=for_training,
                          do_xnlp=do_xnlp,
                          alt_dataset_group=alt_dataset_group)

    print(training_sets.keys())
    print(training_sets["ner"].keys())

    if not for_training or do_xnlp:
        model.reload_mappings()

        char_to_id, id_to_char, id_to_morpho_tag, id_to_tag, id_to_word, \
        morpho_tag_to_id, tag_to_id, word_to_id =\
            extract_mapping_dictionaries_from_model(model)
    else:
        word_to_id, id_to_word, \
        char_to_id, id_to_char, \
        tag_to_id, id_to_tag, \
        morpho_tag_to_id, id_to_morpho_tag = \
            create_mappings(training_sets,
                            parameters,
                            file_format=parameters['file_format'],
                            morpho_tag_separator=("+" if model.parameters['lang_name'] == "turkish" else ud_morpho_tag_separator))

    if opts.overwrite_mappings and for_training:
        print('Saving the mappings to disk...')
        model.save_mappings(id_to_word, id_to_char, id_to_tag, id_to_morpho_tag)

    data_dict = {"ner": {}, "md": {}}
    unique_words_dict = {"ner": {}, "md": {}}
    stats_dict = {"ner": {}, "md": {}}

    # Index data
    if for_training or do_xnlp:
        for label in ["ner", "md"]:
            for purpose in ["train", "dev"]:
                if label in training_sets and purpose in training_sets[label]:
                    _, stats_dict[label][purpose], unique_words_dict[label][purpose], data_dict[label][purpose] = \
                        prepare_dataset(training_sets[label][purpose],
                                        word_to_id, char_to_id, tag_to_id, morpho_tag_to_id,
                                        parameters['lower'], parameters['mt_d'], parameters['mt_t'], parameters['mt_ci'],
                                        file_format=parameters['file_format'],
                                        morpho_tag_separator=("+" if model.parameters['lang_name'] == "turkish" else ud_morpho_tag_separator))

    for label in ["ner", "md"]:
        print(label)
        _, stats_dict[label]["test"], unique_words_dict[label]["test"], data_dict[label]["test"] = \
            prepare_dataset(
                training_sets[label]["test"],
                word_to_id, char_to_id, tag_to_id, morpho_tag_to_id,
                parameters['lower'], parameters['mt_d'], parameters['mt_t'], parameters['mt_ci'],
                file_format=parameters['file_format'],
                morpho_tag_separator=("+" if model.parameters['lang_name'] == "turkish" else ud_morpho_tag_separator))

    if for_training or do_xnlp:
        for label in ["ner", "md"]:
            purposes = ["train", "dev", "test"]
            n_values = sum([1 for purpose in purposes if purpose in stats_dict[label]])
            part1 = " ".join(["{}", "/"] * n_values)
            part2a = " / ".join([purpose for purpose in purposes if purpose in stats_dict[label]])

            for object_name, values_list in \
                    [("sentences", [len(stats_dict[label][purpose]) for purpose in purposes if purpose in stats_dict[label]]),
                    ("words", [sum([x[0] for x in stats_dict[label][purpose]]) for purpose in purposes if purpose in stats_dict[label]]),
                    ("longest sentences", [max([x[0] for x in stats_dict[label][purpose]]) for purpose in purposes if purpose in stats_dict[label]]),
                    ("shortest sentences", [min([x[0] for x in stats_dict[label][purpose]]) for purpose in purposes if purpose in stats_dict[label]])
                                ]:
                part2 = " " + object_name + " in " + part2a + "."
                print(((part1 + part2).format(*values_list)))

            for i, stats_label in [[2,
                                    'char',
                                    ]]:
                for values_list, stats_label_determiner in [
                    [[sum([sum(x[i]) for x in stats_dict[label][purpose]]) for purpose in purposes if purpose in stats_dict[label]],
                     "total {} ".format(stats_label)],
                    [[max([max(x[i]) for x in stats_dict[label][purpose]]) for purpose in purposes if purpose in stats_dict[label]],
                     "max. {} lengths".format(stats_label)],
                    [[min([min(x[i]) for x in stats_dict[label][purpose]]) for purpose in purposes if purpose in stats_dict[label]],
                     "min. {} lengths".format(stats_label)],
                ]:
                    part2 = " " + stats_label_determiner + " in " + part2a + "."
                    print(((part1 + part2).format(*values_list)))

    else:
        for label in ["ner", "md"]:
            purposes = ["dev", "test"]
            n_values = sum([1 for purpose in purposes if purpose in stats_dict[label]])
            part1 = " ".join(["{}", "/"] * n_values)
            part2a = " / ".join([purpose for purpose in purposes if purpose in stats_dict[label]])

            for object_name, values_list in \
                    [("sentences", [len(stats_dict[label][purpose]) for purpose in purposes if purpose in stats_dict[label]]),
                     ("words", [sum([x[0] for x in stats_dict[label][purpose]]) for purpose in purposes if purpose in stats_dict[label]]),
                     ("longest sentences", [max([x[0] for x in stats_dict[label][purpose]]) for purpose in purposes if purpose in stats_dict[label]]),
                     ("shortest sentences", [min([x[0] for x in stats_dict[label][purpose]]) for purpose in purposes if purpose in stats_dict[label]])
                     ]:
                part2 = " " + object_name + " in " + part2a + "."
                print(((part1 + part2).format(*values_list)))

            for i, stats_label in [[2,
                                    'char',
                                    ]]:
                for values_list, stats_label_determiner in [
                    [[sum([sum(x[i]) for x in stats_dict[label][purpose]]) for purpose in purposes if purpose in stats_dict[label]],
                     "total {} ".format(stats_label)],
                    [[max([max(x[i]) for x in stats_dict[label][purpose]]) for purpose in purposes if purpose in stats_dict[label]],
                     "max. {} lengths".format(stats_label)],
                    [[min([min(x[i]) for x in stats_dict[label][purpose]]) for purpose in purposes if purpose in stats_dict[label]],
                     "min. {} lengths".format(stats_label)],
                ]:
                    part2 = " " + stats_label_determiner + " in " + part2a + "."
                    print(((part1 + part2).format(*values_list)))

    print("Max. sentence lengths: %s" % max_sentence_lengths)
    print("Max. char lengths: %s" % max_word_lengths)

    if for_training and not do_xnlp:
        triple_list = []
        for purpose in ["train", "dev", "test"]:
            triple_list += [[purpose, stats_dict["ner"][purpose], unique_words_dict["ner"][purpose]]] if purpose in stats_dict["ner"] else []

        for label, bucket_stats, n_unique_words in triple_list:
            int32_items = len(stats_dict["ner"]["train"]) * (max_sentence_lengths[label] * (5 + max_word_lengths[label]) + 1)
            float32_items = n_unique_words * parameters['word_dim']
            total_size = int32_items + float32_items
            # TODO: fix this with byte sizes
            logging.info("Input ids size of the %s dataset is %d" % (label, int32_items))
            logging.info(
                "Word embeddings (unique: %d) size of the %s dataset is %d" % (n_unique_words, label, float32_items))
            logging.info("Total size of the %s dataset is %d" % (label, total_size))

    # # Save the mappings to disk
    # print 'Saving the mappings to disk...'
    # model.save_mappings(id_to_word, id_to_char, id_to_tag, id_to_morpho_tag)

    # return data_dict, id_to_tag, word_to_id, stats_dict, id_to_char, id_to_morpho_tag

    if for_training:
        # return data_dict, id_to_tag, word_to_id, stats_dict, parameters['t_s']
        return data_dict, id_to_tag, word_to_id, stats_dict, id_to_char, id_to_morpho_tag
        # return dev_data, {}, id_to_tag, parameters['t_s'], test_data, \
        #        train_data, train_stats, word_to_id, yuret_test_data, yuret_train_data
    else:
        # return dev_data, {}, id_to_tag, parameters['t_s'], test_data, [], {}, word_to_id, yuret_test_data, []
        return data_dict, id_to_tag, word_to_id, stats_dict, id_to_char, id_to_morpho_tag


def extract_mapping_dictionaries_from_model(model):
    # words
    # dico_words, word_to_id, id_to_word
    id_to_word = dict(model.id_to_word)
    word_to_id = {word: word_id for word_id, word in list(id_to_word.items())}
    # id_to_word[10000000] = "<UNK>"
    # word_to_id["<UNK>"] = 10000000
    # chars
    id_to_char = dict(model.id_to_char)
    char_to_id = {char: char_id for char_id, char in list(id_to_char.items())}
    # tags
    id_to_tag = dict(model.id_to_tag)
    print(id_to_tag)
    tag_to_id = {tag: tag_id for tag_id, tag in list(id_to_tag.items())}
    print(tag_to_id)
    # morpho_tags
    id_to_morpho_tag = dict(model.id_to_morpho_tag)
    morpho_tag_to_id = {morpho_tag: morpho_tag_id for morpho_tag_id, morpho_tag in list(id_to_morpho_tag.items())}
    return char_to_id, id_to_char, id_to_morpho_tag, id_to_tag, id_to_word, morpho_tag_to_id, tag_to_id, word_to_id


### not used at the moment

def calculate_global_maxes(max_sentence_lengths, max_word_lengths):
    global_max_sentence_length = 0
    global_max_char_length = 0
    for i, d in enumerate([max_sentence_lengths, max_word_lengths]):
        for label in list(d.keys()):
            if i == 0:
                if d[label] > global_max_sentence_length:
                    global_max_sentence_length = d[label]
            elif i == 1:
                if d[label] > global_max_char_length:
                    global_max_char_length = d[label]
    return global_max_sentence_length, global_max_char_length
