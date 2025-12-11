# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the math dataset to parquet format
"""

import os
import datasets
import sys

project_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
sys.path.append(project_dir)
print(project_dir)

from verl.utils.hdfs_io import copy, makedirs
import argparse
from utils import log_dataset

from verl.utils.reward_score.math_dataset import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    os.chdir('..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/dapo')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=7500)
    parser.add_argument('--test_size', type=int, default=5000)

    args = parser.parse_args()

    
    data_source = "open-r1/DAPO-Math-17k-Processed"
    datasource_name = 'DAPO'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    dataset = datasets.load_dataset(data_source, 'all',trust_remote_code=True)["train"]
    log_dataset(dataset)


    math_prompts = [x['prompt'][0]['content'] for x in datasets.load_dataset('parquet',data_files='./data/math/train.parquet')["train"]]
    print(f"lenght of math prompts {len(math_prompts)}")
    def filter_math_prompt(example):
        return not (example['prompt'] in math_prompts)
    print(f"length of dataset {len(dataset)} before filtering")
    dataset = dataset.filter(filter_math_prompt)
    print(f"length of dataset {len(dataset)} after filtering")
    # instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example['prompt']
            solution = example['solution']

            # Only keep necessary fields for training
            data = {
                "data_source": datasource_name,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(solution)  # Ensure string type
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    dataset = dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=dataset.column_names)
    dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
