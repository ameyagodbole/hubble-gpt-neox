#!/usr/bin/env python
# Copyright (c) 2024, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import subprocess
import sys


def main(input_args=None):
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    from megatron.neox_arguments import NeoXArgs
    from megatron.utils import get_wandb_api_key

    neox_args = NeoXArgs.consume_deepy_args(input_args)
    deepspeed_main_args = neox_args.get_deepspeed_main_args()

    # Extract wandb API key and inject into worker environments
    wandb_token = get_wandb_api_key(neox_args=neox_args)
    if wandb_token is not None:
        os.environ["WANDB_API_KEY"] = wandb_token

    # deepspeed.launcher.launch.main(deepspeed_main_args)

    # cmd = ["torchrun", "--nnodes", os.environ["SLURM_JOB_NUM_NODES"],
    #        "--nproc_per_node". os.environ["SLURM_GPUS_ON_NODE"],
    #        "--master-addr", os.environ["MASTER_ADDR"],
    #        "--master-port", os.environ["MASTER_PORT"],
    #        "--node-rank", os.environ["RANK"],
    #        "--log-dir", f"{os.getcwd()}/logs",]
    cmd = ["deepspeed", "--no_ssh", "--node_rank", os.environ["SLURM_LOCALID"]] + deepspeed_main_args
    env = os.environ.copy()
    curr_path = os.path.abspath('.')
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = curr_path + ":" + env['PYTHONPATH']
    else:
        env['PYTHONPATH'] = curr_path

    logging.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.Popen(cmd, env=env)

    result.wait()

    # In case of failure must propagate the error-condition back to the caller (usually shell). The
    # actual error and traceback should have been printed in the subprocess, so in order to avoid
    # unnecessary noise we just quietly exit here with the same code as the subprocess
    if result.returncode > 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
