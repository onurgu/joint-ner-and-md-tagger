import json
import sys

token_index = 1
line = sys.stdin.readline().strip()
while line:

    if line == "\n":
        token_index = 1
        print("")
    else:
        tokens = line.strip().split(" ")

        conllu_tokens = ["_" for i in range(10)]

        conllu_tokens[0] = str(token_index)
        conllu_tokens[1] = tokens[0]

        json_dict = {}

        json_dict["CORRECT_ANALYSIS"] = tokens[1]

        conllu_tokens[5] = tokens[1]

        analyses = tokens[2:-1]  # in Oflazer format, so no =
        json_dict["ALL_ANALYSES"] = analyses
        json_dict["NER_TAG"] = tokens[-1]

        conllu_tokens[-1] = json.dumps(json_dict, separators=(',', ':'))

        print(("\t".join(conllu_tokens)))
        token_index += 1
    line = sys.stdin.readline()
