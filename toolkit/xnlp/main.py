import dynet
from itertools import chain
import numpy as np
import pandas as pd

def concentration(embeddings):
    """

    calculate the concentration of embedding vectors as defined in Language Models Learn POS first

    :type embeddings: list
    :param embeddings:

    :return:
    """
    norms = [0, 0]
    c_array = []
    for vec in embeddings:
        print(vec)
        norms[0] = np.sum(np.abs(vec))
        norms[1] = np.sqrt(np.sum(np.square(vec)))
        c_array.append(norms[1]/norms[0])

    return np.array(c_array).reshape(1, -1)


def test_concentration():

    embeddings = [[1, 0, 0], [0, 1, 0]]

    c_array = concentration(embeddings)

    assert (c_array == np.array([[1, 1]])).all()


def test_dev_obtain_valid_paths():
    from collections import namedtuple

    model = namedtuple('model', ["entity_types"])
    model.entity_types = ["PER", "LOC"]

    from functools import partial
    model._obtain_valid_paths = partial(dev_obtain_valid_paths, model)

    valid_paths = list(model._obtain_valid_paths(4))

    assert len(valid_paths) == -1, valid_paths


def dev_obtain_valid_paths(self, sequence_length):

    if sequence_length == 0:
        # yield []
        pass # do not yield
    elif sequence_length == 1:
        for entity_type in self.entity_types:
            yield ["S-%s" % entity_type]
    else:
        for entity_type in self.entity_types:
            for right_valid_path in self._obtain_valid_paths(sequence_length - 1):
                yield ["S-%s" % entity_type] + right_valid_path
        for l in range(2, sequence_length+1):
            valid_path = [""] * l
            valid_path[0] = "B-%s"
            for i in range(1, l):
                valid_path[i] = "I-%s"
            valid_path[-1] = "E-%s"
            for entity_type in self.entity_types:
                for right_valid_path in self._obtain_valid_paths(sequence_length-l):
                    # yield ["tag1"] + right_valid_path
                    yield [(x % entity_type) for x in valid_path] + right_valid_path
                if l == sequence_length:
                    # yield ["tag2"] + [l, sequence_length]
                    yield [(x % entity_type) for x in valid_path]


import datetime as dt
import linecache
import os
from resource import getrusage, RUSAGE_SELF
import tracemalloc

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def generate_raw_explanations(args, data_dir):
    models = {
        "finnish_model_10_size": ("./xnlp/data/models",
                                 "model-00002218/",
                                 "model-epoch-00000030/"),
        "finnish_model_100_size": ("./models",
                                 "model-00002715/",
                                 "model-epoch-00000047/"),
        "turkish_model_100_size": ("./models",
                                   "model-00002714/",
                                   "model-epoch-00000026/"),
    }

    from utils.evaluation import do_xnlp

    model, data_dict, id_to_tag, word_to_id, stats_dict, id_to_char, id_to_morpho_tag, opts, parameters = \
        do_xnlp(
            *models[args.model_label],
            modify_paths_in_opts=False if args.on_truba else True
        )


    from lib.lime.lime.lime_text import LimeConllSentenceExplainer, ConllSentenceDomainMapper, IndexedConllSentence

    explainer = LimeConllSentenceExplainer(verbose=True, feature_selection="none")

    unique_morpho_tag_types = set(model.id_to_morpho_tag.values())

    morpho_tag_to_id = {k: i for i, k in model.id_to_morpho_tag.items()}

    with open(os.path.join(data_dir, "id_to_morpho_tag-for-ner-train-%s.txt" % args.model_label), "w") as out_id_to_morpho_tag_f:
        out_id_to_morpho_tag_f.write(
            "\t".join([" ".join(map(str, t)) for t in sorted(model.id_to_morpho_tag.items(), key=lambda x: x[0])]) + "\n")

    lime_explanations = []
    raw_explanations = []

    for sample_idx, sample in enumerate(data_dict['ner']['train']):
        max_rss = getrusage(RUSAGE_SELF).ru_maxrss

        print(dt.datetime.now(), 'max RSS', max_rss)

        indexed_conll_sentence = IndexedConllSentence(sample)
        domain_mapper = ConllSentenceDomainMapper(indexed_conll_sentence)
        from utils.evaluation import extract_multi_token_entities
        for entity_start, entity_end, entity_type in extract_multi_token_entities(
                [model.id_to_tag[i] for i in sample['tag_ids']]):
            entity_positions = (entity_start, entity_end)
            # extract the golden labels for the sequence
            entity_tags = [model.id_to_tag[i] for i in
                           sample['tag_ids'][entity_positions[0]:entity_positions[-1]]]

            morpho_tag_types_found_in_the_sample_as_ids = set().union(*[set().union(*[set(morpho_tag_sequence)
                                                                                      for morpho_tag_sequence in
                                                                                      morpho_tag_sequences])
                                                                        for morpho_tag_sequences in
                                                                        sample['morpho_analyzes_tags'][
                                                                        entity_positions[0]:entity_positions[-1]]])

            morpho_tag_types_found_in_the_sample = [model.id_to_morpho_tag[i] for i in
                                                    sorted(list(morpho_tag_types_found_in_the_sample_as_ids))]

            class_names = model.obtain_valid_paths(entity_end - entity_start)
            class_names = [x[1] for x in
                           sorted([(" ".join(class_name), class_name) for class_name in class_names],
                                  key=lambda x: x[0])]
            target_entity_tag_sequence_label_id = class_names.index(entity_tags)

            dynet.renew_cg()
            exp, configurations, probs = explainer.explain_instance(sample,
                                                                    entity_positions,
                                                                    class_names,
                                                                    model.probs_for_a_specific_entity,
                                                                    labels=(target_entity_tag_sequence_label_id,),
                                                                    num_samples=100,
                                                                    num_features=len(
                                                                        morpho_tag_types_found_in_the_sample_as_ids),
                                                                    strategy="NER_TAG_TYPE_REMOVAL",
                                                                    strategy_params_dict={
                                                                        "morpho_tag_types": sorted(
                                                                            list(
                                                                                morpho_tag_types_found_in_the_sample_as_ids)),
                                                                        "n_unique_morpho_tag_types": len(
                                                                            unique_morpho_tag_types),
                                                                        "perturbate_only_entity_indices": True
                                                                    }
                                                                    )

            lime_explanation_summary = "\t".join([str(sample_idx), entity_type, " ".join([str(x) for x in [entity_start, entity_end]])] +
                          [" ".join([x[0], str(x[1])]) for x in domain_mapper.translate_feature_ids_in_exp(
                              exp.local_exp[target_entity_tag_sequence_label_id],
                              morpho_tag_types_found_in_the_sample)]) + "\n"

            print(lime_explanation_summary)
            lime_explanations.append(lime_explanation_summary)

            one_liners = []
            for tmp in [configurations, probs]:
                out_string = ""
                out_string += "%d %d " % tmp.shape
                out_string += " ".join(["%e" % x for x in list(tmp.ravel())])
                one_liners.append(out_string)

            raw_explanation = "\t".join(one_liners
                                              + [str(target_entity_tag_sequence_label_id)]
                                              + [" ".join([str(len(model.id_to_morpho_tag))] + [str(x) for x in sorted(
                                        list(morpho_tag_types_found_in_the_sample_as_ids))])]) + "\n"

            raw_explanations.append(raw_explanation)

    return lime_explanations, raw_explanations


def explain_using_raw_probs(args, data_dir):

    files = {"all": "../../explanations-for-ner-train-finnish-20190114-total.txt",
             "only_target_entities": "../../explanations-for-ner-train-finnish-20190115-total-only_target_entities.txt",
             "finnish_model_10_size": {"explanations": "../../explanations-for-ner-train-finnish_model_10_size.txt",
                                       "raw_data": "../../regression-data-for-ner-train-finnish_model_10_size.txt"},
             "finnish_model_100_size": {"explanations": "explanations-for-ner-train-finnish_model_100_size.txt",
                                        "raw_data": "regression-data-for-ner-train-finnish_model_100_size.txt",
                                        "id_to_morpho_tag": "id_to_morpho_tag-for-ner-train-finnish_model_100_size.txt"},
             "turkish_model_100_size": {"explanations": "explanations-for-ner-train-turkish_model_100_size.txt",
                                        "raw_data": "regression-data-for-ner-train-turkish_model_100_size.txt",
                                        "id_to_morpho_tag": "id_to_morpho_tag-for-ner-train-turkish_model_100_size.txt"}}

    lines = []
    raw_data_records = []
    with open(os.path.join(data_dir, files[args.model_label]["raw_data"]), "r") as f:
        lines = f.readlines()
        for line in lines:
            first_part, second_part, third_part, fourth_part = line.strip().split("\t")

            size_x, size_y, *conf_data = [int(float(x)) for x in first_part.split(" ")]
            C = np.reshape(conf_data, (size_x, size_y))

            size_x, size_y, *probs_data = [float(x) for x in second_part.split(" ")]
            P = np.reshape(probs_data, (int(size_x), int(size_y)))

            target_class_index = int(third_part)

            n_morpho_tags, *morpho_tag_ids = [int(x) for x in fourth_part.split(" ")]

            record = (C, P, target_class_index, n_morpho_tags, morpho_tag_ids)
            raw_data_records.append(record)

    lines = []
    records = []
    with open(os.path.join(data_dir, files[args.model_label]["explanations"]), "r") as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.strip().split("\t")
            record = [int(tokens[0]), tokens[1], tuple([int(x) for x in tokens[2].split(" ")])]
            record.append({k: float(v) for k, v in [tuple(x.split(" ")) for x in tokens[3:]]})
            records.append(record)

    # version without duplicates
    from collections import defaultdict

    zero_centered_Ps = defaultdict(list)
    indexed_Cs = defaultdict(list)
    for i in range(len(raw_data_records)):
        C = raw_data_records[i][0]
        P = raw_data_records[i][1]
        target_class_index = raw_data_records[i][2]
        n_morpho_tags = raw_data_records[i][3]
        morpho_tag_ids_per_sentence = raw_data_records[i][4]
        target_entity_type = records[i][1]

        unperturbated_configuration = [0] * n_morpho_tags
        for morpho_tag_id in morpho_tag_ids_per_sentence:
            unperturbated_configuration[morpho_tag_id] = 1
        #     indexed_C = [0]*n_morpho_tags
        #     for idx in range(len(indexed_C)):
        #         indexed_C[idx] = list(unperturbated_configuration)

        indexed_C = [list(unperturbated_configuration)]
        for idx in range(n_morpho_tags):
            tainted = False
            perturbated_configuration = list(unperturbated_configuration)
            for morpho_tag_idx, morpho_tag_id in enumerate(morpho_tag_ids_per_sentence):
                if idx == morpho_tag_id and unperturbated_configuration[morpho_tag_id] == 1:
                    perturbated_configuration[morpho_tag_id] = -1
                    tainted = True
            if tainted:
                indexed_C.append(perturbated_configuration)
        indexed_Cs[target_entity_type] += [np.array(indexed_C)]

        zero_centered_P = [0.0]
        for morpho_tag_id, diff_value in zip(morpho_tag_ids_per_sentence,
                                             list(P[1:, target_class_index] - P[0, target_class_index])):
            zero_centered_P.append(diff_value)
        zero_centered_Ps[target_entity_type] += [zero_centered_P]

    for target_entity_type in zero_centered_Ps.keys():
        zero_centered_Ps[target_entity_type] = np.array(zero_centered_Ps[target_entity_type])
        indexed_Cs[target_entity_type] = np.array(indexed_Cs[target_entity_type])

    with open(os.path.join(data_dir, files[args.model_label]["id_to_morpho_tag"]), "r") as id_to_morpho_tag_f:
        id_to_morpho_tag = {int(x.split(" ")[0]): x.split(" ")[1] for x in
                            id_to_morpho_tag_f.readline().strip().split("\t")}

    explanations = dict()
    for entity_type in zero_centered_Ps.keys():
        explanations[entity_type] = []
        for sentence_idx in range(indexed_Cs[entity_type].shape[0]):
            from sklearn.linear_model import Ridge

            reg_loc = Ridge(alpha=1, fit_intercept=False)

            cur_X = indexed_Cs[entity_type][sentence_idx]  # (89, 89)
            cur_Y = zero_centered_Ps[entity_type][sentence_idx]  # (89,)
            reg_loc.fit(cur_X, cur_Y)

            # print("sentence: %d, intercept: %lf", sentence_idx, reg_loc.intercept_)

            cur_explanation = sorted([(idx, id_to_morpho_tag[idx], value) for idx, value in
                                      zip(sorted(id_to_morpho_tag.keys()), reg_loc.coef_)],
                                     key=lambda x: x[2],
                                     reverse=True)
            cur_str_explanation = "\n".join(
                [" ".join((str(idx), morpho_tag, "%.7lf" % weight)) for idx, morpho_tag, weight in cur_explanation])
            #     print(cur_explanation)
            explanations[entity_type].append(cur_explanation)

    explanations_nparray_dict = {}
    for entity_type in zero_centered_Ps.keys():
        explanations_nparray_dict[entity_type] = np.array(
            [[t[2] for t in sorted(explanations[entity_type][i], key=lambda x: x[0])] for i in
             range(len(explanations[entity_type]))])

    return indexed_Cs, zero_centered_Ps, id_to_morpho_tag, explanations, explanations_nparray_dict


def generate_tables_in_latex(language_name, zero_centered_Ps, id_to_morpho_tag, explanations_nparray_dict):
    ret_dict = {}
    for entity_type in zero_centered_Ps.keys():
        mean_for_entity_type = sorted(
            [(id_to_morpho_tag[idx], el) for idx, el in enumerate(explanations_nparray_dict[entity_type].mean(axis=0))],
            key=lambda x: x[1], reverse=True)
        zero_means_for_entity_type = [x for x in mean_for_entity_type if x[1] == 0]
        list_to_be_added = [mean_for_entity_type[:10], mean_for_entity_type[-10:],
                            zero_means_for_entity_type]
        list_to_be_added += list(chain.from_iterable(
            [[mean_for_entity_type[:i], mean_for_entity_type[-i:]] for i in range(1, 10)]))
        ret_dict[entity_type] = list_to_be_added
        limited_mean_for_entity_type = mean_for_entity_type[:10] + mean_for_entity_type[-10:]
        df_results = pd.DataFrame([x for x in limited_mean_for_entity_type],
                                  index=[x[0] for x in limited_mean_for_entity_type])
        print("\\begin{table}")
        print(df_results.to_latex(header=["Morphological Tag", "Average Weight"], index=False))
        print("\\caption{Average weights over the corpus for %s %s entities\label{tab:%s_corpus_average}}" % (language_name, entity_type, entity_type.lower()))
        print("\\end{table}")
        print("")
    return ret_dict


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--command", required=True)
    parser.add_argument("--model_label", required=True)
    parser.add_argument("--on_truba", default=False)

    args = parser.parse_args()

    data_dir = "./toolkit/xnlp/"

    if args.command == "generate_explanations":
        lime_explanations, raw_explanations = generate_raw_explanations(args, data_dir)
        with open(os.path.join(data_dir, "explanations-for-ner-train-%s.txt" % args.model_label), "w") as out_f, \
                open(os.path.join(data_dir, "regression-data-for-ner-train-%s.txt" % args.model_label), "w") as regression_data_f:
            for line in lime_explanations:
                out_f.write(line)
            for line in raw_explanations:
                regression_data_f.write(line)
    elif args.command == "explain_using_raw_probs":
        indexed_Cs, zero_centered_Ps, id_to_morpho_tag, explanations, explanations_nparray_dict = \
            explain_using_raw_probs(args, data_dir)
        language_name = args.model_label.split("_")[0]
        language_name = language_name[0].upper() + language_name[1:]
        top_and_bottom_morpho_tags_dict = generate_tables_in_latex(language_name,
                                                                   zero_centered_Ps,
                                                                   id_to_morpho_tag,
                                                                   explanations_nparray_dict)
        from itertools import chain
        for entity_type, top_and_bottom_morpho_tags in top_and_bottom_morpho_tags_dict.items():
            for idx, label in enumerate(["top", "bottom", "zero"] +
                                        list(chain.from_iterable(zip(["top%02d" % i for i in range(1, 10)],
                                                                     ["bottom%02d" % i for i in range(1, 10)])))):
                print("%s_morpho_tags_%s=%s" % (entity_type, label, ",".join([str(x[0]) for x in top_and_bottom_morpho_tags[idx]])))
