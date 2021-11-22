import os
import shutil


class DynetSaver():

    def __init__(self, parameter_collection, checkpoint_dir, max_saves=3):
        self.parameter_collection = parameter_collection
        self.checkpoint_dir = checkpoint_dir
        self.max_saves = max_saves

    def save(self, epoch=None, n_bests=None):
        assert epoch or (n_bests >= 0), "One of epoch or n_bests should be specified"
        subdirs = [dir for dir, _, _ in os.walk(self.checkpoint_dir) if os.path.basename(dir).startswith("model-epoch-")]
        for dir_to_be_deleted in subdirs[:-(self.max_saves-1)]:
            shutil.rmtree(dir_to_be_deleted)
        model_dir_path = "model-epoch-%08d" % epoch if epoch is not None else ("best-models-%08d" % n_bests)
        model_checkpoint_dir_path = os.path.join(self.checkpoint_dir, model_dir_path)
        if not os.path.exists(model_checkpoint_dir_path):
            os.mkdir(model_checkpoint_dir_path)
        self.parameter_collection.save(os.path.join(model_checkpoint_dir_path,
                                                    "model.ckpt"))

    def restore(self, filepath):
        self.parameter_collection.populate(filepath)
