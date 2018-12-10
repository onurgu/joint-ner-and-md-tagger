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

    import IPython
    IPython.embed()