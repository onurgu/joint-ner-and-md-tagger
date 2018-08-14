
import codecs
import os
import subprocess
import tempfile

from utils import tokenizer

analyzer_paths = {'turkish': './utils/analyzers/turkish/'}
analyzer_command = {'turkish': ["./bin/lookup",
                                "-latin1",
                                "-f",
                                "tfeatures.scr"]}


def create_single_word_single_line_format(string_output, conll=False, for_prediction=False):
    lines = string_output.split("\n")
    if not conll:
        result = "<S> <S>+BSTag\n"
    else:
        result = ""
    current_single_line = ""
    subline_idx = 0
    for line in lines:
        if line != "":
            tokens = line.split("\t")
            if subline_idx == 0:
                current_single_line += tokens[0]
                if conll:
                    current_single_line += " " + "_"
                current_single_line += " " + tokens[1] + tokens[2]
            else:
                current_single_line += " " + tokens[1] + tokens[2]
            subline_idx += 1
        else:
            if conll and for_prediction and len(current_single_line) > 0:
                result += current_single_line + " O" + "\n"
            else:
                result += current_single_line + "\n"
            subline_idx = 0
            current_single_line = ""
    if not conll:
        result = result[:-1]
        result += "</S> </S>+ESTag\n"
    return result


def get_morph_analyzes(line, lang="turkish"):
    """

    :param lang:
    :param line:
    :return:
    """
    if type(line) == unicode:
        tokens = tokenizer.tokenize(line)
    else:
        tokens = tokenizer.tokenize(line.decode("utf8"))
    fd, f_path = tempfile.mkstemp()
    with open(f_path, "w") as f:
        for token in tokens:
            f.write(token.encode("iso-8859-9") + "\n")
    os.close(fd)
    print f_path
    with codecs.open(f_path, "r", encoding="iso-8859-9") as f, open(os.devnull, "w") as devnull:
        # print f.readlines()
        string_output = subprocess.check_output(analyzer_command[lang],
                                                stdin=f,
                                                cwd=analyzer_paths[lang],
                                                stderr=devnull)

    # print string_output
    # print type(string_output)
    # print string_output.decode("iso-8859-9").encode('utf8')
    # print type(string_output.decode("iso-8859-9"))
    return string_output
