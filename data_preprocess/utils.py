from datasets import Dataset
def make_prefix(dp):
    problem = dp['problem']
    prefix = f"""Please solve the following math problem: {problem}. The assistant first thinks about the reasoning process step by step and then provides the user with the answer. Return the final answer in \\boxed{{}} tags, for example \\boxed{{1}}. Let's solve this step by step. """
    return prefix

def log_dataset(dataset:Dataset):
    print(f'len:{len(dataset)}')
    first_data = dataset[0]
    print(f'keys: {first_data.keys()}')
    for key in first_data.keys():
        print(f'{key}: {first_data[key]}')