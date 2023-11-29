import os
import json
import torch
import sys
import pickle
from scipy import stats
import argparse

sys.path.append('/data/hoyeon/entity_knowledge_propagation')

from transformers import set_seed
from src import metrics
from src import run_edit
from src import data_utils

ROOT_DIR = '/data/hoyeon/entity_knowledge_propagation'

def main(epoch, args):

    # Choose from 'ft_per_ex', 'ft', 'prepend_def'
    ki_method = 'ft'

    # Pretrained model. Use this together with 'ft'.
    ft_model_name = None
    model_info = args.exp_name

    # Choose a unique experiment name
    #exp_name = f'ecbd/gpt/final/{ki_method}_{model_info}'
    exp_name = f'ecbd/gpt2/final/{ki_method}_{model_info}'

    exp_dir = os.path.join(ROOT_DIR, 'output', exp_name)

    os.makedirs(exp_dir, exist_ok=False)

    data_dir = os.path.join(ROOT_DIR, 'data')

    data_files = [
        # os.path.join(data_dir, 'ecbd/all_ent_2020_2021_np_500samples.json'),
        # os.path.join(data_dir, 'ecbd/all_ent_2020_2021_random_500samples.json')
        os.path.join(data_dir, 'ecbd/all_ent_2020_2021_np_easy.json')
    ]

    train_params = {
        "EXP_NAME": exp_name,
        "EXP_DIR": exp_dir,
        "KI_METHOD": ki_method,
        "BASE_MODEL": args.model_name, #"gpt2-xl",  # model_type: # t5-base/t5-large, "gpt-neo-1.3B"
        "TRAIN_BATCH_SIZE": 1,  # training batch size
        "VALID_BATCH_SIZE": 1,  # validation batch size
        "TRAIN_EPOCHS": 1,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": float(args.lr),  # learning rate 3e-6
        "MAX_SOURCE_TEXT_LENGTH": 128,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 128,  # max length of target text
        "SEED": 2022,  # set seed for reproducibility
        "AUGMENT": False,
        "NUM_AUGMENT": 1,
        "X_IS_Y": False,
        "MASKING_STRATEGY": 'random',
        "USE_NLL_MASKS": False,
        "TOPK": 1,
        "MEMORY_RETRIEVAL": False,
        "TRAIN_ON_PROBE": False,
        "COMPUTE_SPECIFICITY": True,  # Set False if prepend_def
        "FREEZE_LAYERS": False,
        "REG_TYPE": '',
        "ALPHA": 0.0
    }

    random_def = None
    if ki_method == 'random_def':
        random_def = data_utils.get_random_definition()

    # Print params
    for k, v in train_params.items():
        print('{:>24}: {}'.format(k, v))

    device = torch.device("cuda:1")
    set_seed(train_params['SEED'])

    results_dict = run_edit.run_experiment(ki_method,
                                           ft_model_name,
                                           'ecbd',
                                           data_files,
                                           device,
                                           train_params,
                                           random_def=random_def,
                                           oracle_ft=False)

    with open(os.path.join(exp_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results_dict, f)

    with open(os.path.join(exp_dir, 'train_params.json'), 'w') as f:
        json.dump(train_params, f)

    print(f'\n\n####################################################################\nresult saved to {exp_dir}\n####################################################################\n\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--lr', type=str, help='learning rate')
    parser.add_argument('--exp_name', type=str, help='exp name')
    parser.add_argument('--model_name', type=str, help='base model')

    # Parse the arguments
    args = parser.parse_args()

    for epoch in [1]:
        main(epoch, args)
