import codecs
import json
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