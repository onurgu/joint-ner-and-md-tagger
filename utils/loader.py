# coding=utf-8
from functools import partial
import itertools
import json
import os
import re
import codecs

from utils import create_dico, create_mapping, zero_digits
from utils import iob2, iob_iobes

from toolkit.joint_ner_and_md_model import MainTaggerModel

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoaderHelper:

    def __init__(self):

        pass

    def read_datasets(self):
        pass

    def prepare_datasets(self):
        pass