
from collections import defaultdict
import codecs
import json
import os
import sys
import subprocess

script_path = os.path.join(os.getcwd(), os.path.dirname(sys.argv[0]))

try:
    import configparser
    cp = configparser
except ImportError as e:
    print(e)
    import configparser
    cp = configparser

assert len(sys.argv) == 2, "You should input the datasets_dirpath"

# lang_names = ["czech",
#               "finnish",
#               "hungarian",
#               "spanish"]

lang_names = ["finnish",
              "turkish"]


# lang_names = [
#               "spanish",
#               ]

# lang_names = ["turkish"]

datasets_dirpath = sys.argv[1]

ini_filepath_template = "{lang_name}-joint-md-and-ner-tagger.ini"


def remove_morpho_tags_from_node(morpho_analysis, morpho_tags_to_remove, lang_name):
    if lang_name == "turkish":
        separator = "+"
        first_part = ''
        morpho_tags = morpho_analysis.split(separator)[1:]
        return separator.join([first_part] + [morpho_tag for morpho_tag in morpho_tags if morpho_tag not in morpho_tags_to_remove])
    else:
        def fix_BLANK(x):
            if x == '':
                return "*BLANK*"
            else:
                return x
        separator = "|"
        if len(morpho_analysis.split(separator)) == 1:
            morpho_analysis = separator.join([morpho_analysis, "*UNKNOWN*"])
        morpho_tags = list(map(fix_BLANK, morpho_analysis.split(separator)[2:]))

        first_morpho_tag = morpho_analysis.split(separator)[1].split("~")[-1]
        first_part = first_morpho_tag if first_morpho_tag not in morpho_tags_to_remove else ""

        left_part = separator.join([morpho_analysis.split(separator)[0], "~".join(morpho_analysis.split(separator)[1].split("~")[:-1] + [first_part])])
        right_part = separator.join(
            [morpho_tag for morpho_tag in morpho_tags if morpho_tag not in morpho_tags_to_remove])
        return left_part + "|" + right_part


for lang_name in lang_names:

    ini_filepath = ini_filepath_template.format(lang_name=lang_name)

    c_parser = cp.ConfigParser()

    lang_dirpath = os.path.join(datasets_dirpath, lang_name)

    c_parser.read([os.path.join(lang_dirpath, ini_filepath)])

    train_filepaths = []

    sections = "md ner".split(" ")

    dataset_filepath_labels = "train_file dev_file test_file".split(" ")
    related_entity_type_and_top_or_bottom_labels = [(x[0], x[1].split(",")) for x in list(c_parser.items('general')) if "_morpho_tags_" in x[0]]
    related_entity_type_and_top_and_bottom_labels = defaultdict(list)
    for x in list(c_parser.items('general')):
        if (x[0].endswith("_morpho_tags_top") or x[0].endswith("_morpho_tags_bottom")):
            key = "_".join(x[0].split("_")[:-1] + ["TOPandBOTTOM"])
            related_entity_type_and_top_and_bottom_labels[key] += x[1].split(",")

    related_entity_type_and_top_or_bottom_labels += [(key, value) for key, value in related_entity_type_and_top_and_bottom_labels.items()]

    for section in sections:
        for dataset_filepath_label in dataset_filepath_labels:
            try:
                dataset_filepath_conllu = os.path.join(lang_dirpath, c_parser.get(section, dataset_filepath_label))

                if section in ["ner", "md"]:
                    # TODO
                    # use the output all_analyses file, get the last column, parse it as a json object, then take another line
                    # from the conllu_file, parse it as a json object, combine them
                    # replace the last column with the combined json, write it to a temporary file, then when finished, rename it
                    all_analyses_filepath = dataset_filepath_conllu + ".all_analyses"
                    if section == "ner":
                        all_analyses_filepath += ".tagged"
                    for related_entity_type_and_top_or_bottom_label, morpho_tags_to_remove in related_entity_type_and_top_or_bottom_labels:
                        with codecs.open(all_analyses_filepath, "r", encoding="utf-8") as output_with_ner_and_md_tags_f, \
                            codecs.open(all_analyses_filepath + ".%s" % related_entity_type_and_top_or_bottom_label, "w",
                                        encoding="utf-8") as output_with_ner_and_md_tags_removed_f:
                            conllu_line_with_ner_and_md_tags = \
                                [f.readline() for f in [output_with_ner_and_md_tags_f]][0]
                            while conllu_line_with_ner_and_md_tags:
                                if conllu_line_with_ner_and_md_tags.startswith("#"):
                                    # comment line
                                    output_with_ner_and_md_tags_removed_f.write(conllu_line_with_ner_and_md_tags)
                                elif conllu_line_with_ner_and_md_tags.strip():
                                    last_column = conllu_line_with_ner_and_md_tags.strip().split("\t")[-1]
                                    if last_column == "_":
                                        output_with_ner_and_md_tags_removed_f.write(conllu_line_with_ner_and_md_tags)
                                    else:
                                        dict_with_ner_and_md_tags = json.loads(last_column)
                                        dict_with_ner_and_md_tags_removed = dict(dict_with_ner_and_md_tags)

                                        for first_level_tag in dict_with_ner_and_md_tags_removed:
                                            node = dict_with_ner_and_md_tags_removed[first_level_tag]
                                            if first_level_tag == "ALL_ANALYSES":
                                                new_node = []
                                                for morph_analysis in list(node):
                                                    new_node.append(remove_morpho_tags_from_node(morph_analysis, morpho_tags_to_remove, lang_name))
                                            elif first_level_tag == "CORRECT_ANALYSIS":
                                                new_node = remove_morpho_tags_from_node(node, morpho_tags_to_remove, lang_name)
                                            else:
                                                new_node = node
                                            dict_with_ner_and_md_tags_removed[first_level_tag] = new_node

                                        cols = conllu_line_with_ner_and_md_tags.strip().split("\t")
                                        cols[-1] = json.dumps(dict_with_ner_and_md_tags_removed)

                                        output_with_ner_and_md_tags_removed_f.write("\t".join(cols)+"\n")
                                else:
                                    output_with_ner_and_md_tags_removed_f.write("\n")
                                conllu_line_with_ner_and_md_tags = \
                                    [f.readline() for f in [output_with_ner_and_md_tags_f]][0]

            except configparser.NoOptionError as e:
                print(e)
