import pickle as pkl
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import argparse
import statistics
import numpy as np


def remove_outliers_iqr(data, multiplier=2.0):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    # print(f"{len(data)-len(filtered_data)}")
    return filtered_data


def load_json(path):
    with open(path) as f:
        return [json.loads(l.strip()) for l in f]


def group_examples(data):
    data_d = defaultdict(list)
    for ex in data:
        ent_id = '_'.join(ex['ex_id'].split('_')[:2])
        data_d[ent_id].append(ex)
    return list(data_d.items())


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


def measure_ppl_drop(per_exs, measure_indices, exclude_pop, remove_outliers=True):
    assert len(per_exs)==1
    per_ex = per_exs[0]
    avg_ppl_drop_per_ex = []
    avg_ppl_fluc_stdev_per_ex = []
    avg_ppl_fluc_abs_per_ex = []
    overall_ppl_drop_per_ex = []
    forget_ratio_per_ex = []
    
    for idx, data in enumerate(per_ex.items()):
        if 'pop' in data[0] and exclude_pop:
            continue
        if idx in measure_indices and idx < 114:
            ppl_drop_on_train = []
            ppl_fluc_not_on_train = []
            forget_ratio = []
            train_idx = data[1]['trained_at']
            ppl = data[1]['ppl']
            cache = None

            for step in range(len(ppl)):
                if step in train_idx:
                    if step!=0:
                        ppl_drop_on_train.append((1-ppl[step]/ppl[step-1])*100)
                        if cache:
                            forget_ratio.append((ppl[step-1]/cache-1)*100)
                        cache = ppl[step]
                        
                else:
                    if step!=0:
                        ppl_fluc_not_on_train.append((1-ppl[step]/ppl[step-1])*100)
            
            # ppl_drop_on_train = remove_outliers_iqr(ppl_drop_on_train)
            if remove_outliers:
                ppl_fluc_not_on_train = remove_outliers_iqr(ppl_fluc_not_on_train)
                ppl_drop_on_train = remove_outliers_iqr(ppl_drop_on_train)
                forget_ratio = remove_outliers_iqr(forget_ratio)
                
            avg_ppl_drop_per_ex.append(sum(ppl_drop_on_train)/len(ppl_drop_on_train))
            avg_ppl_fluc_stdev_per_ex.append(statistics.pstdev(ppl_fluc_not_on_train))
            avg_ppl_fluc_abs_per_ex.append(sum([abs(num) for num in ppl_fluc_not_on_train])/len(ppl_fluc_not_on_train))
            overall_ppl_drop_per_ex.append((1-ppl[-1]/ppl[0])*100)
            forget_ratio_per_ex.append(sum(forget_ratio)/len(forget_ratio))
        
    avg_ppl_drop_on_train = sum(avg_ppl_drop_per_ex)/len(avg_ppl_drop_per_ex)
    avg_ppl_fluc_stdev_not_on_train = sum(avg_ppl_fluc_stdev_per_ex)/len(avg_ppl_fluc_stdev_per_ex)
    avg_ppl_fluc_abs_not_on_train = sum(avg_ppl_fluc_abs_per_ex)/len(avg_ppl_fluc_abs_per_ex)
    avg_forget_ratio = sum(forget_ratio_per_ex)/len(forget_ratio_per_ex)
    
    if remove_outliers:
        overall_ppl_drop_per_ex = remove_outliers_iqr(overall_ppl_drop_per_ex)
        
    avg_overall_ppl_drop = sum(overall_ppl_drop_per_ex)/len(overall_ppl_drop_per_ex)
    
    result = {
        'ppl_drop_on_train': (avg_ppl_drop_on_train, avg_ppl_drop_per_ex),
        'ppl_fluc_stdev_not_on_train': (avg_ppl_fluc_stdev_not_on_train, avg_ppl_fluc_stdev_per_ex),
        'ppl_fluc_abs_not_on_train': (avg_ppl_fluc_abs_not_on_train, avg_ppl_fluc_abs_per_ex),
        'forget_ratio': (avg_forget_ratio, forget_ratio_per_ex),
        'overall_ppl_drop': (avg_overall_ppl_drop, overall_ppl_drop_per_ex) 
        }
    
    
    return result


def main(args):

    # Filtered samples
    measure_indices = [
            1, 3, 5, 7, 8, 9, 11, 15, 16, 17, \
            20, 22, 23, 25, 26, 27, 29, 30, 31, 32, \
            34, 37, 39, 40, 41, 43, 44, 45, 46, 47, \
            49, 51, 53, 54, 55, 59, 60, 62, 65, 66, \
            70, 71, 73, 76, 82, 83, 84, 85, 86, 89, \
            91, 92, 97, 98, 103, 106, 110, 112, 113] # 59 examples

    # Process data_file
    train_dataset = load_json(args.train_data)
    valid_dataset = load_json(args.valid_data)
    # print(len(dataset))
    ex_ids = []
    for d in valid_dataset:
        ex_ids.append(d['ex_id'])
    group = group_examples(train_dataset)
    # print(len(group))
    
    # Process train_idx
    train_idx = get_train_idx(train_dataset)
    # print(len(train_idx))

    # Process results.pkl files
    results_list = [load_results(args.out_dir, exp_name) for exp_name in args.exp_name]

    # Rearrange PPL data
    per_exs = []
    for results in results_list:
        
        per_ex = {ex_id: {'ppl': [], 'trained_at': []} for ex_id in ex_ids}
        # print(len(per_ex))

        for step, result in enumerate(results):
            # print(step)
            spec_data = result['specificity']
            for spec_idx, ppl_data in enumerate(spec_data):
                # print(spec_idx)
                per_ex[ex_ids[spec_idx]]['ppl'].append(ppl_data['post'][0])
            per_ex[result['ex_id']]['trained_at'].append(step)

        # print([ex['trained_at'] for ex in per_ex.values()])

        for ent, ex in per_ex.items():
            new_ppl = []
            new_trained_at = [find_bin(train_idx, ex['trained_at'][0])+n*len(group) for n in range(5)] if len(ex['trained_at'])!=0 else []
            count = 0
            for idx, ppl in enumerate(ex['ppl']):
                if idx%len(train_dataset) in train_idx:
                    new_ppl.append(ppl)
                # print(idx, count)
                # print(new_trained_at)
                    
            ex['ppl'] = new_ppl
            ex['trained_at'] = new_trained_at
            # print(len(new_ppl))
            assert len(new_ppl)%65==0
            # assert len(new_trained_at)==5
        # print(per_ex)
        # print(train_idx)
        per_exs.append(per_ex)


    if args.mode=='draw_figures':
        plot_indices = list(range(len(per_ex)))
        # plot_indices = train_idx
        
        figure_path = os.path.join('figures/', args.save_dir)
        os.makedirs(figure_path, exist_ok=True)
        
        for i, key in enumerate(tqdm(per_exs[0])):
            if i in plot_indices:
                plot_ppl_with_trained_at(per_exs, key, args.exp_name, filename=f'{figure_path}/{i}.png')
    
    
    elif args.mode=='measure_scores':
        # measure_indices = list(range(len(per_ex)))
        result = measure_ppl_drop(per_exs, measure_indices, exclude_pop=True)
        # assert len(avg_ppl_drop_per_ex)==len(measure_indices)
        print(f"\n\n################################################################################\n \
                avg_ppl_drop_per_ex:\n\n{result['ppl_drop_on_train'][1]}\n\n \
                avg_ppl_drop: {result['ppl_drop_on_train'][0]} \
                \n\navg_ppl_fluc_abs_per_ex:\n\n{result['ppl_fluc_abs_not_on_train'][1]}\n\n \
                avg_ppl_fluc_abs: {result['ppl_fluc_abs_not_on_train'][0]}\n\n \
                \n\navg_ppl_fluc_stdev_per_ex:\n\n{result['ppl_fluc_stdev_not_on_train'][1]}\n\n \
                avg_ppl_fluc_stdev: {result['ppl_fluc_stdev_not_on_train'][0]}\n\n \
                \n\navg_forget_ratio_per_ex:\n\n{result['forget_ratio'][1]}\n\n \
                avg_forget_ratio: {result['forget_ratio'][0]}\n\n \
                overall_ppl_drop: {result['overall_ppl_drop'][0]} \
                \n################################################################################\n\n")
        
    
    elif args.mode=='order_examples':
        
        def sort_idx(lst):
            sorted_pairs = sorted(zip(measure_indices, lst), key=lambda x: x[1], reverse=True)
            return [index for index, value in sorted_pairs]
        
        result = measure_ppl_drop(per_exs, measure_indices, exclude_pop=True, remove_outliers=False)  
        
        sort_by_ppl_drop = sort_idx(result['ppl_drop_on_train'][1])
        sort_by_fluc_abs = sort_idx(result['ppl_fluc_abs_not_on_train'][1])
        sort_by_fluc_stdev = sort_idx(result['ppl_fluc_stdev_not_on_train'][1])
        sort_by_forget_ratio = sort_idx(result['forget_ratio'][1])
        sort_by_overall_ppl_drop = sort_idx(result['overall_ppl_drop'][1])
        
        print(f"\n\n################################################################################\n \
                    avg_ppl_drop_per_ex:\n\n{sort_by_ppl_drop}\n\n \
                    \n\navg_ppl_fluc_abs_per_ex:\n\n{sort_by_fluc_abs}\n\n \
                    \n\navg_ppl_fluc_stdev_per_ex:\n\n{sort_by_fluc_stdev}\n\n \
                    \n\navg_forget_ratio_per_ex:\n\n{sort_by_forget_ratio}\n\n \
                    overall_ppl_drop: {sort_by_overall_ppl_drop} \
                    \n################################################################################\n\n")
    
    else:
        raise NotImplementedError
        
        
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    # out_dir = "./output/ecbd/gpt2/final/"
    # exp_name = "ft_medium_8e-6"
    # data_file = "./data/ecbd/all_ent_2020_2021_np_easy.json"


    # Add arguments
    parser.add_argument('--out_dir', type=str, default="./output/ecbd/gpt2/final/")
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--exp_name', nargs='+', required=True)
    parser.add_argument('--train_data', type=str, default="./data/ecbd/all_ent_2020_2021_np_easy.json")
    parser.add_argument('--valid_data', type=str, default="./data/ecbd/validation.json")
    parser.add_argument('--mode', type=str, default="draw_figures")

    # Parse the arguments
    args = parser.parse_args()

    main(args)