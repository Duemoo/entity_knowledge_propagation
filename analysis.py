import pickle as pkl
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import argparse


def load_json(path):
    with open(path) as f:
        return [json.loads(l.strip()) for l in f]


def group_examples(data):
    data_d = defaultdict(list)
    for ex in data:
        ent_id = '_'.join(ex['ex_id'].split('_')[:2])
        data_d[ent_id].append(ex)
    return list(data_d.items())


def find_idx(ex_ids, ex_id):
    return ex_ids.index(ex_id)


def get_train_idx(data):
    group = group_examples(data)
    # print(group[2])
    train_indices = []
    count = 0
    for i, item in enumerate(group):
        # print(item)
        for instance_idx, instance in enumerate(item[1]):
            if instance_idx==0:
                train_indices.append(count)
            count += 1
    return train_indices


def reorder_results(results):
    pass


def plot_ppl_with_trained_at(per_exs, key, exp_names, filename='plot.png'):
    
    
    ppls = [per_ex[key]['ppl'] for per_ex in per_exs]
    trained_at_indices = [per_ex[key]['trained_at'] for per_ex in per_exs]

    # plt.figure(figsize=(10, 6))
    # fig, axs = plt.subplots(1, len(per_exs), figsize=(5*len(per_exs), 5)) 
    # plt.plot(ppl, label='ppl values')
    
    # Mark the trained_at indices
    plt.figure(figsize=(len(per_exs)*8, 5))
    for i in range(len(per_exs)):
        plt.subplot(1, len(per_exs), i+1)
        
        ppl, trained_at_index = ppls[i], trained_at_indices[i]
        plt.plot(ppl, label='ppl values')
        
        for idx in trained_at_index:
            if idx < len(ppl):
                plt.scatter(idx, ppl[idx], color='red', s=10)
                plt.text(idx, ppl[idx], f' {idx}', verticalalignment='bottom')
            if idx != 0:
                plt.scatter(idx-1, ppl[idx-1], color='red', s=10)
                plt.text(idx-1, ppl[idx-1], '', verticalalignment='bottom')

        plt.xlabel('Training Step')
        plt.ylabel('ppl Value')
        plt.title(f'{exp_names[i]}')
        plt.legend()
        plt.grid(True)

    # Save the figure to a file
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    

def find_bin(train_idx, num):
    for i in range(len(train_idx[:-1])):
        if num>=train_idx[i] and num<train_idx[i+1]:
            return i
    return len(train_idx)-1


def load_results(out_dir, exp_name):
    result_file = os.path.join(args.out_dir, exp_name, 'results.pkl')
    with open(result_file, 'rb') as f:
        results = pkl.load(f)
    try:
        results = results['/data/hoyeon/entity_knowledge_propagation/data/ecbd/all_ent_2020_2021_np_easy.json']
    except:
        results = results['/workspace/entity_knowledge_propagation/data/ecbd/all_ent_2020_2021_np_easy.json'] 
    return results


def main(args):

    # Process data_file
    dataset = load_json(args.data_file)
    ex_ids = []
    for d in dataset:
        ex_ids.append(d['ex_id'])
    group = group_examples(dataset)
    
    # Process train_idx
    train_idx = get_train_idx(dataset)

    # Process results.pkl files
    results_list = [load_results(args.out_dir, exp_name) for exp_name in args.exp_name]

    # Rearrange PPL data
    per_exs = []
    for results in results_list:
        
        per_ex = {ex_id: {'ppl': [], 'trained_at': []} for ex_id in ex_ids}

        for step, result in enumerate(results):
            spec_data = result['specificity']
            for spec_idx, ppl_data in enumerate(spec_data):
                per_ex[ex_ids[spec_idx]]['ppl'].append(ppl_data['post'][0])
                
            per_ex[result['ex_id']]['trained_at'].append(step)

        for ent, ex in per_ex.items():
            new_ppl = []
            new_trained_at = [find_bin(train_idx, ex['trained_at'][0])+n*65 for n in range(5)]
            count = 0
            for idx, ppl in enumerate(ex['ppl']):
                if idx%len(dataset) in train_idx:
                    new_ppl.append(ppl)
                # print(idx, count)
                # print(new_trained_at)
                    
            ex['ppl'] = new_ppl
            ex['trained_at'] = new_trained_at
            assert len(new_ppl)%65==0
            assert len(new_trained_at)==5
        # print(per_ex)
        # print(train_idx)
        per_exs.append(per_ex)

    plot_indices = list(range(len(per_ex)))
    # plot_indices = train_idx
    
    figure_path = os.path.join('figures/', args.save_dir)
    os.makedirs(figure_path, exist_ok=True)
    
    for i, key in enumerate(tqdm(per_exs[0])):
        if i in plot_indices:
            plot_ppl_with_trained_at(per_exs, key, args.exp_name, filename=f'{figure_path}/{i}.png')
        
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # out_dir = "./output/ecbd/gpt2/final/"
    # exp_name = "ft_medium_8e-6"
    # data_file = "./data/ecbd/all_ent_2020_2021_np_easy.json"


    # Add arguments
    parser.add_argument('--out_dir', type=str, default="./output/ecbd/gpt2/final/")
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--exp_name', nargs='+', required=True)
    parser.add_argument('--data_file', type=str, default="./data/ecbd/all_ent_2020_2021_np_easy.json")

    # Parse the arguments
    args = parser.parse_args()

    main(args)