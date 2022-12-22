
import codecs
import os
import subprocess
import sys
import tempfile

from utils import tokenizer

analyzer_paths = {'turkish': './utils/analyzers/turkish/'}
analyzer_command = {'turkish': ["./bin.linux64/lookup",
                                "-latin1",
                                "-f",
                                "tfeatures.scr"]}


def turkish_lower(s):
    return s.replace("IİŞÜĞÖÇ", "ıişüğöç")


def create_single_word_single_line_format(morph_analyzer_output_for_a_single_sentence,
                                          conll=False, for_prediction=False):
    """

    Transform Oflazer's analyzer's output into single line output format with morphological analyzes

    :param morph_analyzer_output_for_a_single_sentence:
    :param conll:
    :param for_prediction:
    :return:
    """
    lines = morph_analyzer_output_for_a_single_sentence.split("\n")
    print(lines)
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
                current_single_line += " " + turkish_lower(tokens[1]).lower() + tokens[2]
            else:
                current_single_line += " " + turkish_lower(tokens[1]).lower() + tokens[2]
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
    print(result)
    return result


def get_morph_analyzes(line, lang="turkish"):
    """

    :param lang:
    :param line: a sentence on a line (untokenized)
    :return:
    """
    if type(line) == str:
        tokens = tokenizer.tokenize(line)
    else:
        tokens = tokenizer.tokenize(line.decode("utf8"))
    fd, f_path = tempfile.mkstemp()
    with codecs.open(f_path, "w", encoding="iso-8859-9", errors="ignore") as f:
        for token in tokens:
            f.write(token + "\n")
    os.close(fd)
    print(f_path)
    with codecs.open(f_path, "r", encoding="iso-8859-9") as f, open(os.devnull, "w") as devnull:
        # print f.readlines()
        string_output = subprocess.check_output(analyzer_command[lang],
                                                stdin=f,
                                                cwd=analyzer_paths[lang],
                                                stderr=devnull)

    print(string_output.decode("iso-8859-9"))
    # print type(string_output)
    # print string_output.decode("iso-8859-9").encode('utf8')
    # print type(string_output.decode("iso-8859-9"))
    return string_output.decode("iso-8859-9")
