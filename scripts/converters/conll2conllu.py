import re
import sys

token_index = 1
line = sys.stdin.readline().strip()
while line:

    if line == "\n":
        token_index = 1
        print("")
    else:
        tokens = re.sub(r"\s+", " ", line.strip().replace("\t", " ")).split(" ")

        conllu_tokens = ["_" for i in range(10)]

        conllu_tokens[0] = str(token_index)
        conllu_tokens[1] = tokens[0]

        conllu_tokens[-1] = "NER_TAG=" + tokens[-1]

        print("\t".join(conllu_tokens))
        token_index += 1
    line = sys.stdin.readline()