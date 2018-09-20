import codecs
import subprocess
import sys

assert len(sys.argv) == 3, \
    "You need to supply two command line arguments: model file path and input file in CONLLU format"

model_filepath=sys.argv[1]
input_filepath=sys.argv[2]

# command_template = "/Users/onur/Desktop/projects/research/projects/focus/morphology/udpipe/cmake-build-debug/udpipe " \
#                    "/Users/onur/Desktop/projects/research/projects/focus/morphology/data/hungarian-ud-2.0-170801.udpipe " \
#                    "/Users/onur/Desktop/projects/research/datasets/hungarian/UD_Hungarian-Szeged/hu_szeged-ud-test.conllu " \
#                    "--input=conllu --tag --tagger=provide_all_analyses --output=conllu " \
#                    "--outfile=denem"

UDPIPE_COMMAND_PATH="/Users/onur/Desktop/projects/research/projects/focus/morphology/udpipe/cmake-build-debug/udpipe"

command_template = "{UDPIPE_COMMAND_PATH} " \
                   "{model_filepath} " \
                   "{input_filepath} " \
                   "--input=conllu --tag --tagger=provide_all_analyses --output=conllu " \
                   "--outfile={output_filepath}"

# print(command_template.format(UDPIPE_COMMAND_PATH=UDPIPE_COMMAND_PATH,
#                                                 model_filepath=model_filepath,
#                                                 input_filepath=input_filepath,
#                                                 output_filepath=input_filepath+".all_analyses").split(" "))

result = subprocess.check_output(command_template.format(UDPIPE_COMMAND_PATH=UDPIPE_COMMAND_PATH,
                                                model_filepath=model_filepath,
                                                input_filepath=input_filepath,
                                                output_filepath=input_filepath+".all_analyses").split(" "))

print(result)

with codecs.open(input_filepath+".all_analyses", "r") as f, codecs.open(input_filepath+".all_analyses.correct_analysis", "w") as  out_f:
    line = f.readline()
    while line:
        output_line = line.strip()
        if line[0] != "#" and len(line.strip()) > 0:
            tokens = line.strip().split("\t")
            if len(tokens) > 0:
                escaped_correct_tags = tokens[5].replace("|", "&").replace("=", ">")
                tokens[-1] += "|CORRECT_ANALYSIS=" + "".join([tokens[2], "&~", tokens[3], "~", tokens[4], "~", escaped_correct_tags])
                output_line = "\t".join(tokens)
        print(output_line)
        out_f.write(output_line + "\n")
        line = f.readline()