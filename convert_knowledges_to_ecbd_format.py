import pickle
import re
import json
from src.data_utils import load_json, format_gpt2_data

knowledge_path = 'generated_knowledge.pkl'

with open(knowledge_path, "rb") as f:
    data=pickle.load(f)

def extract(sentence, is_probe):
    if is_probe:
        pattern = r':\s(.*?)\s_'
    else:
        pattern = r':\s(.*)'
    match = re.search(pattern, sentence)
    if match:
        extracted_text = match.group(1)
    return extracted_text.strip()


instances = []
for i, k in enumerate(data):
    try:
        split_k = k.split('\n')
        split_k_cleaned = [s for s in split_k if s]
        # print(split_k_cleaned)
            
        definition = extract(split_k_cleaned[0], is_probe=False)
        probe = extract(split_k_cleaned[1], is_probe=True)
        target = extract(split_k_cleaned[2], is_probe=False)
        # print(f"\n\n{target}\n\n")
        
        instance = {
            "ex_id": f"knowledge_{i}",
            "definition": definition + "<extra_id_0>",
            "def_target": "<extra_id_0>  <extra_id_1>",
            "qid": [f"knowledge_{i}"], 
            "ent_str": f"knowledge_{i}", 
            "attribute": "n/a", 
            "additional_sentences": [], 
            "conceptnet": [], 
            "probe_sentences": {"template_0": {"probe_sentence": probe + " <extra_id_0>", "label": f"<extra_id_0> {target} <extra_id_1>"}}
        }
        instances.append(instance)
    # data example:
    # {"ex_id": "Inter-Parliamentary Alliance on China_64177105_4_7", 
    # "definition": "The Inter-Parliamentary Alliance on China (IPAC) is an international<extra_id_0> alliance of parliamentarians from democratic countries focused on relations with the People's Republic of China (PRC), and specifically, the Chinese Communist Party (CCP).", 
    # "def_target": "<extra_id_0> , cross-party <extra_id_1>", 
    # "category": "tv_show", 
    # "qid": ["international organization (Q484652)"], 
    # "ent_str": "Inter-Parliamentary Alliance on China", 
    # "attribute": "n/a", 
    # "additional_sentences": [], 
    # "conceptnet": [], 
    # "probe_sentences": 
    #   {"template_0": 
    #          {"probe_sentence": "On 22 September 2020, speaking as the co-chair of the Inter-Parliamentary Alliance on China, Duncan Smith made a more assertive statement asking the IOC to think again about allowing <extra_id_0> to           hosting the games, addressing China with the words: \"The free world does have a strong position to say the bullying, the threatening, the internal repression, the border disputes, the arrogant attitude to your              neighbours, the breaking of the treaty with Hong Kong \u2014 these must have consequences.\"", 
    #           "label": "<extra_id_0> China <extra_id_1>"}
    #   }
    # }
    
    # print(f"def: {definition}\nprobe: {probe}\ntarget: {target}")
    
    except:
        print(f'exception occurred at instance {i}')
        continue


# ref = load_json('data/ecbd/all_ent_2020_2021_np_easy.json')[1]
# print('\n\n\n')
# print(format_gpt2_data(instances[0]))
# print('\n\n\n')
# print(format_gpt2_data(ref))
with open('custom_knowledges_200.json', 'w') as f:
    for instance in instances:
        json.dump(instance, f)
        f.write('\n')