#!/usr/bin/env bash
# Copyright (c) 2020-2021 The MMSegmentation Authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


CONFIG=$1
GPUS=$2
PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
