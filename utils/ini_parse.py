import argparse


try:
    # Python 2
    import ConfigParser
    cp = ConfigParser
except ImportError as e:
    # print(e)
    import configparser
    cp = configparser

parser = argparse.ArgumentParser("ini_parse.py")

parser.add_argument("--input", required=True)

parser.add_argument("--only_values", default=False, action="store_true")

parser.add_argument("--add_suffixes", default=False, action="store_true")

parser.add_argument("--query", nargs="*")

args = parser.parse_args()


c_parser = cp.ConfigParser()


c_parser.read([args.input])


result_str = []
for query in args.query:
    tokens = query.split(".")

    try:
        dataset_filepath = c_parser.get(tokens[0], tokens[1])
        if args.add_suffixes:
            if tokens[0] == "ner":
                dataset_filepath += ".all_analyses.tagged"
            elif tokens[0] == "md":
                dataset_filepath += ".all_analyses"
        if not args.only_values:
            dataset_filepath = "_".join(tokens) + "=" + dataset_filepath

        # print(dataset_filepath)
        result_str.append(dataset_filepath)
    except configparser.NoOptionError as e:
        pass

print((" ".join(result_str)))
# print(args)