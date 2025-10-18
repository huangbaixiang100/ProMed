import json
import os
from multiprocessing import Pool, Manager
from tqdm import tqdm
import numpy as np

from mcts import *

# Corresponds to the SIG-Guided MCTS Sampling process
def process_mcts(data):
    """ operate MCTS search, return the best path and score """
    #if 'shapley' not in data.keys():
    #    return data
    mcts = MCTS(data, benchmark='medqa',iterations=5,
        process_reward_fn=get_llm_evalute_score,
        use_correctness_reward=False
    )
    mcts.search()
    all_paths = mcts.export_paths(mcts.root, mcts.custom_reward)

    paths = []
    for path, reward in all_paths:
        states = []
        for node in path:
            states.append(node.state)
        path_data = {
            'reward': reward,
            'dialogue': path[-1].doctor_dialogue,
            'states': states
        }
        paths.append(path_data)
    data['all_paths'] = paths

    return data

def worker(args):
    """ process a single data case, run MCTS and save the results """
    data, out_path, lock = args

    result = process_mcts(data)

    if 'all_paths' not in result.keys() or not result['all_paths']:
        return False


    with lock:
        with open(out_path, 'a', encoding='utf-8') as outfile:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

    return True


if __name__ == "__main__":
    out_path = "test.jsonl"

    with open("softmax_shapley.jsonl", 'r', encoding='utf-8') as f:
        id_to_shapley = {}
        for line in f:
            record = json.loads(line)
            shapley = record['shapley']
            weights = normalize_shapley_weights(np.array(shapley), method="softmax", temperature=2.0)
            id_to_shapley[record['id']] = weights.tolist()

    # read the original data and merge with Shapley values
    file_name = '../data/dataset/medqa_atomic_data_example.jsonl'
    with open(file_name, 'r', encoding='utf-8') as f:
        if 'jsonl' in file_name:
            datas = []
            for line in f:
                datas.append(json.loads(line))
        else:
            datas = json.load(f)

    for item in datas:
        data_id = item['id']
        if data_id in id_to_shapley:
            item['shapley'] = id_to_shapley[data_id]


    # filter out already predicted data
    processed_ids = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding='utf-8') as f:
            for line in f:
                processed_ids.add(json.loads(line)['id'])

    test_datas = [data for data in datas if data["id"] not in processed_ids]

    manager = Manager()
    lock = manager.Lock()

    with Pool(100) as p:
        for result in tqdm(p.imap_unordered(worker, [(data, out_path, lock) for data in test_datas]),
                           total=len(test_datas)):
            if not result:
                print("Error in processing data")
                with open('log.txt', 'a') as log_file:
                    log_file.write("Error during processing\n")
