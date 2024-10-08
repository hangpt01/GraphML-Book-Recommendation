import numpy as np 
import pandas as pd 
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Process cf data
def read_cf(file_name):
    df = pd.read_csv(file_name)
    df[['user_id', 'item_id']] = df[['patron_record_num', 'item_record_num']]
    df_ = df[['user_id', 'item_id']]
    train_df, test_df = train_test_split(df_, test_size=0.2, random_state=68)
    return np.array(train_df.values), np.array(test_df.values)


def read_triplets(file_name):
    can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
    can_triplets_np = np.unique(can_triplets_np, axis=0)
    return can_triplets_np


def process_cf_data(in_dir, out_dir):
    train_np, test_np = read_cf(in_dir+ '/library.csv') 
    with open(out_dir + '/train.txt', 'w+') as f:
        for line in train_np:
            f.write(f"{line[0]}\t{line[1]}\t1\n")
    with open(out_dir + '/test.txt', 'w+') as f:
        for line in test_np:
            f.write(f"{line[0]}\t{line[1]}\t1\n")

# Process KG data
def process_kg_data(in_dir, out_dir, dataset):
    kg_data = read_triplets(in_dir +'/kg_final.txt')
    
    with open(out_dir + f'/{dataset}.kg', 'w+') as f:
        f.write('head_id:token\trelation_id:token\ttail_id:token\n')
        for line in kg_data:
            f.write(f"{line[0]}\t{line[1]}\t{line[2]}\n")

if  __name__ == "__main__":
    dataset= 'library'
    in_dir = f'./dataset/{dataset}'
    out_dir = f'./dataset/{dataset}'
    # process_cf_data(in_dir,out_dir)
    process_kg_data(in_dir, out_dir, dataset)