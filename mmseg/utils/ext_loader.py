# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib


def load_ext(name, funcs):
    ext = importlib.import_module('mmseg.' + name)
    for fun in funcs:
        assert hasattr(ext, fun), f'{fun} miss in module {name}'

    return ext
