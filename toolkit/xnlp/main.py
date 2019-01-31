import dynet
import numpy as np


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



if __name__ == "__main__":



    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_label", required=True)
    parser.add_argument("--on_truba", default=False)

    args = parser.parse_args()

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

    # tracemalloc.start()

    from lib.lime.lime.lime_text import LimeConllSentenceExplainer, ConllSentenceDomainMapper, IndexedConllSentence

    explainer = LimeConllSentenceExplainer(verbose=True, feature_selection="none")

    unique_morpho_tag_types = set(model.id_to_morpho_tag.values())

    morpho_tag_to_id = {k: i for i, k in model.id_to_morpho_tag.items()}

    with open("explanations-for-ner-train-%s.txt" % args.model_label, "w") as out_f:

        for sample_idx, sample in enumerate(data_dict['ner']['train']):
            max_rss = getrusage(RUSAGE_SELF).ru_maxrss
            # snapshot = tracemalloc.take_snapshot()
            print(dt.datetime.now(), 'max RSS', max_rss)
            # display_top(snapshot)
            indexed_conll_sentence = IndexedConllSentence(sample)
            domain_mapper = ConllSentenceDomainMapper(indexed_conll_sentence)
            from utils.evaluation import extract_multi_token_entities
            for entity_start, entity_end, entity_type in extract_multi_token_entities([model.id_to_tag[i] for i in sample['tag_ids']]):
                entity_positions = (entity_start, entity_end)
                # extract the golden labels for the sequence
                entity_tags = [model.id_to_tag[i] for i in
                               sample['tag_ids'][entity_positions[0]:entity_positions[-1]]]

                morpho_tag_types_found_in_the_sample_as_ids = set().union(*[set().union(*[set(morpho_tag_sequence)
                                                                                          for morpho_tag_sequence in
                                                                                          morpho_tag_sequences])
                                                                            for morpho_tag_sequences in
                                                                            sample['morpho_analyzes_tags'][entity_positions[0]:entity_positions[-1]]])

                morpho_tag_types_found_in_the_sample = [model.id_to_morpho_tag[i] for i in
                                                        sorted(list(morpho_tag_types_found_in_the_sample_as_ids))]

                class_names = model.obtain_valid_paths(entity_end-entity_start)
                class_names = [x[1] for x in
                               sorted([(" ".join(class_name), class_name) for class_name in class_names], key=lambda x: x[0])]
                target_entity_tag_sequence_label_id = class_names.index(entity_tags)

                dynet.renew_cg()
                exp = explainer.explain_instance(sample,
                                                 entity_positions,
                                                 class_names,
                                                 model.probs_for_a_specific_entity,
                                                 labels=(target_entity_tag_sequence_label_id,),
                                                 num_samples=100,
                                                 num_features=len(morpho_tag_types_found_in_the_sample_as_ids),
                                                 strategy="NER_TAG_TYPE_REMOVAL",
                                                 strategy_params_dict={
                                                         "morpho_tag_types": sorted(
                                                             list(morpho_tag_types_found_in_the_sample_as_ids)),
                                                         "n_unique_morpho_tag_types": len(unique_morpho_tag_types),
                                                         "perturbate_only_entity_indices": True
                                                        }
                                                 )

                # print(domain_mapper.translate_feature_ids_in_exp(exp.local_exp[target_entity_tag_label_id],
                #                                                  morpho_tag_types_found_in_the_sample))
                # print(" ".join(
                #     [x[0] for x in domain_mapper.translate_feature_ids_in_exp(exp.local_exp[target_entity_tag_label_id],
                #                                                               morpho_tag_types_found_in_the_sample)][:10]))
                # sorted_by_feature_name = sorted(
                #     domain_mapper.translate_feature_ids_in_exp(exp.local_exp[target_entity_tag_label_id],
                #                                                morpho_tag_types_found_in_the_sample),
                #     key=lambda x: x[0])
                # print(sorted_by_feature_name)

                print("\t".join([str(sample_idx), entity_type, " ".join([str(x) for x in [entity_start, entity_end]])] +
                    [" ".join([x[0], str(x[1])]) for x in domain_mapper.translate_feature_ids_in_exp(exp.local_exp[target_entity_tag_sequence_label_id],
                                                                                                     morpho_tag_types_found_in_the_sample)]))
                out_f.write("\t".join([str(sample_idx), entity_type, " ".join([str(x) for x in [entity_start, entity_end]])] +
                    [" ".join([x[0], str(x[1])]) for x in domain_mapper.translate_feature_ids_in_exp(exp.local_exp[target_entity_tag_sequence_label_id],
                                                                                                     morpho_tag_types_found_in_the_sample)]) + "\n")
                out_f.flush()