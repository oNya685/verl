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
import sys

project_dir = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
sys.path.append(project_dir)
print(project_dir)

from verl.utils.hdfs_io import copy, makedirs
import argparse
import json
from datasets import Dataset


def load_dataset():
    dataset_dir = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/luxiaodong06/OFRL/dataset"
    train_data = [{'problem': x[0]['value'], 'solution': x[1]['ground_truth']['value']}
                for x in json.load(open(f'{dataset_dir}/orz_math_57k_collected.json', 'r'))]
    train_data = Dataset.from_list(train_data)
    print(train_data[5]['problem'])
    print(train_data[8]['solution'])
    return train_data

if __name__ == '__main__':
    os.chdir('..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/orz')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()
    data_source_name = 'ORZ'
    dataset = load_dataset()

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question = example['problem']
            solution = example['solution']
            
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

    dataset = dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=dataset.column_names)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    # print data source and length
    print(f"Length of train dataset: {len(dataset)}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)



    