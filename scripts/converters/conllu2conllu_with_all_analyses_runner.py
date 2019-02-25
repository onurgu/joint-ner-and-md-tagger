
import codecs
import json
import os
import sys
import subprocess

script_path = os.path.join(os.getcwd(), os.path.dirname(sys.argv[0]))

try:
    import configparser
    cp = configparser.ConfigParser
except ImportError as e:
    print(e)
    import configparser
    cp = configparser

assert len(sys.argv) == 2, "You should input the datasets_dirpath"

lang_names = ["czech",
              "finnish",
              "hungarian",
              "spanish"]

# lang_names = [
#               "spanish",
#               ]

# lang_names = ["turkish"]

datasets_dirpath = sys.argv[1]

ini_filepath_template = "{lang_name}-joint-md-and-ner-tagger.ini"

for lang_name in lang_names:
    ini_filepath = ini_filepath_template.format(lang_name=lang_name)

    c_parser = cp.ConfigParser()

    lang_dirpath = os.path.join(datasets_dirpath, lang_name)

    c_parser.read([os.path.join(lang_dirpath, ini_filepath)])

    model_path = c_parser.get("udpipe", "model")

    train_filepaths = []

    # ner_train_filepath = c_parser.get("ner", "train_file")
    #
    # print(model_path, ner_train_filepath)

    command_template = "python " \
                       "{script_path}/conllu2conllu_with_all_analyses.py " \
                       "{model_path} " \
                       "{input_filepath}"

    sections = "md ner".split(" ")

    dataset_filepath_labels = "train_file dev_file test_file".split(" ")

    for section in sections:
        for dataset_filepath_label in dataset_filepath_labels:
            try:
                dataset_filepath_conllu = os.path.join(lang_dirpath, c_parser.get(section, dataset_filepath_label))
                command = command_template.format(script_path=script_path,
                                                  model_path=os.path.join(lang_dirpath, model_path),
                                                  input_filepath=dataset_filepath_conllu)
                print(command)
                subprocess.check_output(command.split(" "))
                if section == "ner":
                    # TODO
                    # use the output all_analyses file, get the last column, parse it as a json object, then take another line
                    # from the conllu_file, parse it as a json object, combine them
                    # replace the last column with the combined json, write it to a temporary file, then when finished, rename it
                    all_analyses_filepath = dataset_filepath_conllu + ".all_analyses"
                    with codecs.open(dataset_filepath_conllu, "r", encoding="utf-8") as input_conllu_f,\
                            codecs.open(all_analyses_filepath, "r", encoding="utf-8") as input_conllu_with_md_tags_f,\
                            codecs.open(all_analyses_filepath + ".tagged", "w", encoding="utf-8") as output_with_ner_and_md_tags_f:
                        conllu_line_with_ner_tags, conllu_line_with_md_tags = \
                            [f.readline() for f in [input_conllu_f, input_conllu_with_md_tags_f]]
                        while conllu_line_with_ner_tags:
                            if conllu_line_with_ner_tags.strip():
                                dict_with_ner_tags, dict_with_md_tags = \
                                    [json.loads(col)
                                    for col in [line.strip().split("\t")[-1]
                                                               for line in [conllu_line_with_ner_tags, conllu_line_with_md_tags]]]
                                dict_with_ner_and_md_tags = dict(dict_with_ner_tags)
                                dict_with_ner_and_md_tags.update(dict_with_md_tags)
                                _tokens = conllu_line_with_md_tags.strip().split("\t")
                                _tokens[-1] = json.dumps(dict_with_ner_and_md_tags, separators=(',', ':'))
                                output_with_ner_and_md_tags_f.write("\t".join(_tokens)+"\n")
                            else:
                                output_with_ner_and_md_tags_f.write("\n")
                            conllu_line_with_ner_tags, conllu_line_with_md_tags = \
                                [f.readline() for f in [input_conllu_f, input_conllu_with_md_tags_f]]

            except configparser.NoOptionError as e:
                print(e)
