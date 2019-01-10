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

if __name__ == "__main__":
    from utils.evaluation import do_xnlp

    model, data_dict, id_to_tag, word_to_id, stats_dict, id_to_char, id_to_morpho_tag, opts, parameters = \
        do_xnlp(
            "./xnlp/data/models",
            "model-00002218/",
            "model-epoch-00000030/"
        )

    from lib.lime.lime.lime_text import LimeConllSentenceExplainer, ConllSentenceDomainMapper, IndexedConllSentence

    explainer = LimeConllSentenceExplainer(verbose=True)

    unique_morpho_tag_types = set(model.id_to_morpho_tag.values())

    morpho_tag_to_id = {k: i for i, k in model.id_to_morpho_tag.items()}

    with open("explanations-for-ner-train-01.txt", "w") as out_f:

        for sample_idx, sample in enumerate(data_dict['ner']['train']):
            indexed_conll_sentence = IndexedConllSentence(sample)
            domain_mapper = ConllSentenceDomainMapper(indexed_conll_sentence)
            from utils.evaluation import extract_multi_token_entities
            for entity_start, entity_end, entity_type in extract_multi_token_entities([model.id_to_tag[i] for i in sample['tag_ids']]):
                entity_positions = (entity_start, entity_end)
                entity_tags = [model.id_to_tag[i] for i in
                               sample['tag_ids'][entity_positions[0]:entity_positions[-1]]]

                morpho_tag_types_found_in_the_sample_as_ids = set().union(*[set().union(*[set(morpho_tag_sequence)
                                                                                          for morpho_tag_sequence in
                                                                                          morpho_tag_sequences])
                                                                            for morpho_tag_sequences in
                                                                            sample['morpho_analyzes_tags']])

                morpho_tag_types_found_in_the_sample = [model.id_to_morpho_tag[i] for i in
                                                        sorted(list(morpho_tag_types_found_in_the_sample_as_ids))]

                class_names = list(model._obtain_valid_paths(entity_end-entity_start))
                class_names = [x[1] for x in
                               sorted([(" ".join(class_name), class_name) for class_name in class_names], key=lambda x: x[0])]
                target_entity_tag_label_id = class_names.index(entity_tags)

                dynet.renew_cg()
                exp = explainer.explain_instance(sample,
                                                 entity_positions,
                                                 class_names,
                                                 model.probs_for_a_specific_entity,
                                                 labels=range(len(class_names)),
                                                 num_samples=100,
                                                 num_features=len(morpho_tag_types_found_in_the_sample_as_ids),
                                                 strategy="NER_TAG_TYPE_REMOVAL",
                                                 strategy_params_dict={
                                                     "morpho_tag_types": sorted(
                                                         list(morpho_tag_types_found_in_the_sample_as_ids)),
                                                     "n_unique_morpho_tag_types": len(unique_morpho_tag_types)}
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
                    [" ".join([x[0], str(x[1])]) for x in domain_mapper.translate_feature_ids_in_exp(exp.local_exp[target_entity_tag_label_id],
                                                                              morpho_tag_types_found_in_the_sample)]))
                out_f.write("\t".join([str(sample_idx), entity_type, " ".join([str(x) for x in [entity_start, entity_end]])] +
                    [" ".join([x[0], str(x[1])]) for x in domain_mapper.translate_feature_ids_in_exp(exp.local_exp[target_entity_tag_label_id],
                                                                              morpho_tag_types_found_in_the_sample)]) + "\n")
                out_f.flush()