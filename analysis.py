import pickle as pkl
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict


out_dir = "./output/ecbd/gpt2/final/"
exp_name = "ft_medium_8e-6"
data_file = "./data/ecbd/all_ent_2020_2021_np_easy.json"
result_file = os.path.join(out_dir, exp_name, 'results.pkl')


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


def plot_ppl_with_trained_at(data, filename='plot.png'):
    ppl = data['ppl']
    trained_at_indices = data['trained_at']

    plt.figure(figsize=(10, 6))
    plt.plot(ppl, label='ppl values')
    
    # Mark the trained_at indices
    for idx in trained_at_indices:
        if idx < len(ppl):
            plt.scatter(idx, ppl[idx], color='red')
            plt.text(idx, ppl[idx], f' {idx}', verticalalignment='bottom')

    plt.xlabel('Index')
    plt.ylabel('ppl Value')
    plt.title('Plot of ppl Values with trained_at Indices Marked')
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


with open(result_file, 'rb') as f:
    results = pkl.load(f)

data_file = load_json(data_file)
train_idx = get_train_idx(data_file)

ex_ids = []
for d in data_file:
    ex_ids.append(d['ex_id'])
group = group_examples(data_file)

try:
    results = results['/data/hoyeon/entity_knowledge_propagation/data/ecbd/all_ent_2020_2021_np_easy.json']
except:
    results = results['/workspace/entity_knowledge_propagation/data/ecbd/all_ent_2020_2021_np_easy.json']
# print(results)
per_ex = {ex_id: {'ppl': [], 'trained_at': []} for ex_id in ex_ids}

print(train_idx)



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
        if idx%len(data_file) in train_idx:
            new_ppl.append(ppl)
        # print(idx, count)
        # print(new_trained_at)
            
    ex['ppl'] = new_ppl
    ex['trained_at'] = new_trained_at
    assert len(new_ppl)%65==0
    assert len(new_trained_at)==5
# print(per_ex)
# print(train_idx)

plot_indices = list(range(len(per_ex)))
# plot_indices = train_idx
os.makedirs(os.path.join('figures', exp_name), exist_ok=True)
for i, key in enumerate(tqdm(per_ex)):
    if i in plot_indices:
        plot_ppl_with_trained_at(per_ex[key], filename=f'figures/{exp_name}/{i}.png')