
import os
import sys
import subprocess

script_path = os.path.join(os.getcwd(), os.path.dirname(sys.argv[0]))

try:
    import ConfigParser
    cp = ConfigParser
except ImportError as e:
    print(e)
    import configparser
    cp = configparser

assert len(sys.argv) == 2, "You should input the datasets_dirpath"

# lang_names = ["czech",
#               "finnish",
#               "hungarian",
#               "spanish",
#               "turkish"]

lang_names = ["turkish"]

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

    sections = "md".split(" ")

    dataset_filepath_labels = "train_file dev_file test_file".split(" ")

    for section in sections:
        for dataset_filepath_label in dataset_filepath_labels:
            try:
                dataset_filepath = c_parser.get(section, dataset_filepath_label)
                command = command_template.format(script_path=script_path,
                                                  model_path=os.path.join(lang_dirpath, model_path),
                                                  input_filepath=os.path.join(lang_dirpath, dataset_filepath))
                print(command)
                subprocess.check_output(command.split(" "))
            except ConfigParser.NoOptionError as e:
                print(e)
