import pickle as pkl
import os
import json


def load_json(path):
    with open(path) as f:
        return [json.loads(l.strip()) for l in f]

out_dir = "./output/ecbd/gpt2/final/"
# exp_name = "ft_ecbd_easy_gpt2-large"
data_file = "./data/ecbd/all_ent_2020_2021_np_easy.json"
exp_name = "ft_ecbd_easy_gpt2-large"
result_file = os.path.join(out_dir, exp_name, 'results.pkl')


ex_ids = []

data_file = load_json(data_file)
for d in data_file:
    ex_ids.append(d['ex_id'])
    

def find_idx(ex_ids, ex_id):
    return ex_ids.index(ex_id)



with open(result_file, 'rb') as f:
    data = pkl.load(f)
    
data = data['/workspace/entity_knowledge_propagation/data/ecbd/all_ent_2020_2021_np_easy.json']

per_ex = {ex_id: {'ppl': [], 'trained_at': []} for ex_id in ex_ids}
for step, instance in enumerate(data):
    spec_data = instance['specificity']
    for spec_idx, ppl_data in enumerate(spec_data):
        per_ex[ex_ids[spec_idx]]['ppl'].append(ppl_data['post'][0])
        
    per_ex[instance['ex_id']]['trained_at'].append(step)

assert all([len(per_ex[ex_id]['ppl'])==len(ex_ids) for ex_id in ex_ids])
assert all([len(per_ex[ex_id]['trained_at'])>0 for ex_id in ex_ids]) 
print(per_ex)