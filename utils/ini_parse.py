import argparse


try:
    import ConfigParser
    cp = ConfigParser
except ImportError as e:
    print(e)
    import configparser
    cp = configparser

parser = argparse.ArgumentParser("ini_parse.py")

parser.add_argument("--input", required=True)

parser.add_argument("--query", nargs="*")

args = parser.parse_args()


c_parser = cp.ConfigParser()


c_parser.read([args.input])


result_str = []
for query in args.query:
    tokens = query.split(".")

    try:
        dataset_filepath = "_".join(tokens) + "=" + c_parser.get(tokens[0], tokens[1])

        # print(dataset_filepath)
        result_str.append(dataset_filepath)
    except ConfigParser.NoOptionError as e:
        pass

print(" ".join(result_str))
# print(args)