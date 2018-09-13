import sys

token_index = 0
line = sys.stdin.readline().strip()
while line:

    if line == "\n":
        token_index = 0
        print("")
    else:
        tokens = line.strip().split(" ")

        conllu_tokens = ["_" for i in range(10)]

        conllu_tokens[0] = str(token_index)
        conllu_tokens[1] = tokens[0]

        conllu_tokens[-1] = "CORRECT_ANALYSIS=" + tokens[1]
        analyses = tokens[2:-1] # in Oflazer format, so no =
        conllu_tokens[-1] += "|ALL_ANALYSES=" + "&".join(analyses)
        conllu_tokens[-1] += "|NER_TAG=" + tokens[-1]

        print("\t".join(conllu_tokens))
        token_index += 1
    line = sys.stdin.readline()
