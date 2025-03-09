from simcse import SimCSE
import json
import numpy as np
import os
import faiss
import random
import sys
import argparse
from itertools import combinations
import ast


def get_parser():
    # Create an argument parser to handle command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="model_path", default="/data/qinglong/GPT-NER-test/models/acge_text_embedding")
    parser.add_argument("--data_path", type=str, help="data_path", default="/data/qinglong/GPT-NER-multi-step-v2/openai_access/low_resource_data/scale_0620/1/test")
    parser.add_argument("--knn_data_path", type=str, help="example_file", default="/data/qinglong/GPT-NER-multi-step-v2/openai_access/low_resource_data/scale_0620/1/knn")
    parser.add_argument("--result_path", type=str, help="example_sentence_file", default="/data/qinglong/GPT-NER-multi-step-v2/openai_access/low_resource_data/scale_0620/1/result")
    parser.add_argument("--train_type", type=str, help="example_num", default="train_all")
    parser.add_argument("--test_type", type=str, help="example_num", default="test_2")
    parser.add_argument("--knn_num", type=int, help="knn_num", default=20)
    parser.add_argument("--test_prefix", type=str, help="knn_num", default="test.2")
    parser.add_argument("--train_prefix", type=str, help="knn_num", default="train_dev.json")
    parser.add_argument("--random", type=bool, default=False)
    return parser

def read_feature(dir_, prefix):
    # Read the feature info file in JSON format
    info_file = json.load(open(os.path.join(dir_, f"{prefix}.start_word_feature_info.json")))
    # Read the feature array using np.memmap
    features = np.memmap(os.path.join(dir_, f"{prefix}.start_word_feature.npy"), 
                         dtype=np.float32,
                         mode="r",
                         shape=(info_file["entity_num"], info_file["hidden_size"]))
    index_file = []
    # Open the feature index file
    file = open(os.path.join(dir_, f"{prefix}.start_word_feature_index.json"), "r")
    for line in file:
        # Convert each line to an integer and append to the index list
        index_file.append(int(line.strip()))
    file.close()
    return info_file, features, index_file

def read_mrc_data(dir_, prefix):
    # Join the directory and prefix to get the file name
    file_name = os.path.join(dir_, f"{prefix}")
    with open(file_name, "r") as f:
        # Read each line, strip newline characters, parse as JSON, and store in a list
        data = [json.loads(line.strip("\n")) for line in f.readlines()]
    return data


def get_example_sentence(example_ids, train_data):
    # Initialize an empty list to store example sentences
    examples = []
    for example_id in example_ids:
        example = []
        # Get the text of each example ID from the training data
        example = [train_data[id]["text"] for id in example_id]
        examples.append(example)
    return examples

def generate_categories(entity_types):
    # Generate all possible non-empty combinations of entity types
    all_combinations = {}
    # Sort the entity types to ensure consistency
    sorted_entity_types = sorted(entity_types)
    for r in range(1, len(sorted_entity_types)+1):
        for combo in combinations(sorted_entity_types, r):
            all_combinations['+'.join(combo)] = []
    return all_combinations

def classify_data(data, entity_types):
    # Generate a classification dictionary
    categories = generate_categories(entity_types)
    categories_index = {}
    for key in categories.keys():
        categories_index[key] = []
    for i, item in enumerate(data):
        # Extract all entity types of the current data item
        types = {label[2] for label in item['label']}
        # Classify the data item based on the combination of entity types
        key = '+'.join(sorted(types))
        if key in categories:
            categories[key].append(item["text"])
            categories_index[key].append(i)
    return categories, categories_index

def read_type_data(file):
    # Read type data from a file
    with open(file, 'r', encoding='utf-8') as f:
        data = f.readlines()
    newdata = []
    for i, d in enumerate(data):
        if d.strip("\n") != "" or d.startswith("{"):
            newdata.append(list(set(list(ast.literal_eval(d.strip("\n"))))))
    return newdata

def compute_simcse_knn(test_mrc_data, train_mrc_data, model_path, knn_num, types_data, test_index=None,  label_map={"量表": "Scale", "测量项目": "Item","测量概念":"Concept"}):
    # Compute KNN for test data using SimCSE model
    sim_model = SimCSE(model_path)
    entity_types = label_map.values()
    train_sentence, train_sentence_index = classify_data(train_mrc_data, entity_types)
    train_index = {}
    for key, _ in train_sentence.items():
        embeddings = sim_model.encode(train_sentence[key], batch_size=128, normalize_to_unit=True, return_numpy=True)
        quantizer = faiss.IndexFlatIP(embeddings.shape[1])
        index = quantizer
        index.add(embeddings.astype(np.float32))
        index.nprobe = min(10, len(train_sentence[key]))
        index = faiss.index_gpu_to_cpu(index)
        train_index[key] = index
    example_idx = []
    example_value = []
    if test_index is None:
        for idx_ in range(len(test_mrc_data)):
            context = test_mrc_data[idx_]["text"]
            types = types_data[idx_]
            if types == ["null"]:
                example_idx.append([])
                example_value.append([])
            else:
                types = [label_map[t] for t in types]
                key = '+'.join(sorted(types))
                embedding = sim_model.encode([context], batch_size=128, normalize_to_unit=True, keepdim=True, return_numpy=True)
                top_value, top_index = train_index[key].search(embedding.astype(np.float32), knn_num)
                example_idx.append([train_sentence_index[key][int(i)] for i in top_index[0]])
                example_value.append([float(value) for value in top_value[0]])
        return example_idx, example_value
    for idx_, sub_index in enumerate(test_index): 
        if sub_index != 0:
            continue
        context = test_mrc_data[idx_]["text"]
        types = types_data[idx_]
        types = [label_map[t] for t in types]
        key = '+'.join(sorted(types))
        embedding = sim_model.encode([context], batch_size=128, normalize_to_unit=True, keepdim=True, return_numpy=True)
        top_value, top_index = train_index[key].search(embedding.astype(np.float32), knn_num)
        example_idx.append([train_sentence_index[key][int(i)] for i in top_index[0]])
        example_value.append([float(value) for value in top_value[0]])
    return example_idx, example_value

def random_knn(test_mrc_data, train_mrc_data, types_data,  knn_num, label_map={"量表": "Scale", "测量项目": "Item","测量概念":"Concept"}):
    # Compute random KNN for test data
    entity_types = label_map.values()
    train_sentence, train_sentence_index = classify_data(train_mrc_data, entity_types)
    example_idx = []
    example_sentence = []
    for idx_ in range(len(test_mrc_data)):
        context = test_mrc_data[idx_]["text"]
        types = types_data[idx_]
        if types == ["null"]:
            example_idx.append([])
            example_sentence.append([])
        else:
            types = [label_map[t] for t in types]
            key = '+'.join(sorted(types))
            random.shuffle(train_sentence_index[key])
            example_idx.append(train_sentence_index[key][:knn_num])
            example_sentence.append([train_mrc_data[i]["text"] for i in example_idx[-1]])
    return example_idx, example_sentence

def write_file(dir_, data):
    # Write data to a file in JSON format
    file = open(dir_, "w")
    for item in data:
        file.write(json.dumps(item, ensure_ascii=False)+'\n')
    file.close()

if __name__ == '__main__':
    # Parse command line arguments
    args = get_parser().parse_args()
    print(args)
    print(args.random)
    # Read test and train MRC data
    test_mrc_data = read_mrc_data(dir_=args.data_path, prefix=args.test_prefix)
    train_mrc_data = read_mrc_data(dir_=args.data_path, prefix=args.train_prefix)
    # Read type data
    types_data = read_type_data(os.path.join(args.result_path, f"tmp.zhipu.type.{args.test_prefix}"))
    if args.random:
        # Compute random KNN if random flag is set
        index_, value_ = random_knn(test_mrc_data=test_mrc_data, train_mrc_data=train_mrc_data, types_data=types_data, knn_num=args.knn_num)
    else:
        # Compute SimCSE KNN if random flag is not set
        index_, value_ = compute_simcse_knn(test_mrc_data=test_mrc_data, train_mrc_data=train_mrc_data, model_path=args.model_path, knn_num=args.knn_num, types_data=types_data)
    # Generate file names for KNN results
    example_file = f"{args.test_type}.simcse.{args.train_type}.{args.knn_num}.knn.json"
    example_sentence_file = f"{args.test_type}.simcse.{args.train_type}.{args.knn_num}.knn.sentence.json"
    # Write KNN indices to file
    write_file(dir_=os.path.join(args.knn_data_path, example_file), data=index_)
    # Get example sentences based on KNN indices
    examples = get_example_sentence(index_, train_mrc_data) 
    # Write example sentences to file
    write_file(dir_=os.path.join(args.knn_data_path, example_sentence_file), data=examples)