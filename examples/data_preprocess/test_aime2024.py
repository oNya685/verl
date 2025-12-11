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
import numpy as np
import sys

project_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
sys.path.append(project_dir)
print(project_dir)

from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math_dataset import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

if __name__ == '__main__':
    os.chdir('..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/aime24')
    parser.add_argument('--hdfs_dir', default=None)
    # parser.add_argument('--train_size', type=int, default=7500)
    # parser.add_argument('--test_size', type=int, default=5000)
    # parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'huggingface.co/datasets/HuggingFaceH4/aime_2024'
    data_source_name = 'AIME24'

    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    test_dataset = dataset['train']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example['problem']
            solution = example['answer']

            # Only keep necessary fields for training
            data = {
                "data_source": data_source_name,
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

    test_dataset = test_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=test_dataset.column_names)
    repeat_test_dataset = test_dataset.repeat(16)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    repeat_test_dataset.to_parquet(os.path.join(local_dir, 'test_16.parquet'))
    # print length of processed data
    print(f"Length of processed data: {len(test_dataset)}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
