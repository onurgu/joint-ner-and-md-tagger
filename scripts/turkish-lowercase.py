#!/usr/bin/env python
# coding=utf8

import sys

CHARMAP = {
    "to_upper": {
        "ı": "I",
        "i": "İ",
    },
    "to_lower": {
        "I": "ı",
        "İ": "i",
    }
}


def lower(s):
    for key, value in list(CHARMAP.get("to_lower").items()):
        s = s.replace(key, value)

    return s.lower()

line = sys.stdin.readline()

while line:
    print(lower(line.decode("utf8").strip()))
    line = sys.stdin.readline()