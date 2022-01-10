# coding=utf-8

import os

import tornado.ioloop
import tornado.web

from utils import read_args
from utils.evaluation import initialize_model_with_pretrained_parameters
from utils.train import models_path


class DisambiguationHandler(tornado.web.RequestHandler):

    model = None

    def initialize(self, model_path, model_epoch_path):

        if DisambiguationHandler.model is None:

            print("loading")
            model, _, _ = initialize_model_with_pretrained_parameters(model_path,
                                                                      model_epoch_path,
                                                                      models_path)
            print("loaded")

            DisambiguationHandler.model = model

    @staticmethod
    def disambiguate_line(line, show_gold_labels):

        from utils.evaluation import predict_sentences_given_model

        print(line)

        labeled_sentences, dataset_file_string = predict_sentences_given_model(line, DisambiguationHandler.model)

        tagger_output_dict = {}
        for i, line in enumerate(labeled_sentences['ner']['test']):
            if len(line) > 0:
                tokens = line.split(" ")
                if not show_gold_labels:
                    # remove the second token, which is the tag
                    tagger_output_dict[i] = [tokens[0], tokens[2]]
                else:
                    tagger_output_dict[i] = tokens

        return {
            'dataset_file_string': {i: line.split(" ") for i, line in enumerate(dataset_file_string.split("\n")) if len(line) > 0},
            'tagger_output': tagger_output_dict
        }
            # 'disambiguator_output': {i: {'surface_form': surface_form, 'analysis': analysis} for i, (surface_form, analysis) in enumerate(prediction_lines_raw)}}


    @tornado.web.asynchronous
    def post(self):

        self.add_header("Access-Control-Allow-Origin", "*")
        line = self.get_argument("textarea", default="Dünyaya hoş geldiniz.")
        show_gold_labels = int(self.get_argument("show_gold_labels", default="0")) == 1
        # print type(line)
        print((line.encode("utf8")))

        line = line.strip()
        self.write(DisambiguationHandler.disambiguate_line(line, show_gold_labels))
        self.write("\n")
        self.finish()


def make_app(opts):
    return tornado.web.Application([
        (r"/ner/predict/", DisambiguationHandler, dict(model_path=opts.model_path, model_epoch_path=opts.model_epoch_path)),
        (r"/(.*)", tornado.web.StaticFileHandler, {"path": os.path.join(os.path.curdir, "./web/public_html/")})
    ])


def start_webapp(sys_argv):

    from utils import read_args

    opts = read_args(args_as_a_list=sys_argv[1:])

    assert type(opts.port) == int

    print("Creating app object")
    app = make_app(opts)
    print("Listening")
    app.listen(opts.port)
    print("Starting the loop")
    tornado.ioloop.IOLoop.current().start()