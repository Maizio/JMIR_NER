import os
from tqdm import tqdm
from base_access import AccessBase
from logger import get_logger
import json
import argparse
from dataset_name import FULL_DATA
import random
import sys
import ast

# Set the random seed to ensure reproducibility of randomness in the code.
random.seed(1)
# Get the logger for the current module.
logger = get_logger(__name__)

# # If you need to set proxy environment variables, you can uncomment the following two lines.
# os.environ["http_proxy"] = "http://localhost:10808"
# os.environ["https_proxy"] = "http://localhost:10809"

# Define a function to get the command line argument parser.
def get_parser():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--strategy", type=str, help="directory for the input")
    # Add various command line arguments for specifying the model, number of examples, data path, etc.
    parser.add_argument("--bigmodel", type=str, help="file name for the input", default="zhipu")
    parser.add_argument("--example_num", type=int, help="file name for the training set", default=3)
    parser.add_argument("--data_path", type=str, help="dataset name for the input", default="/data/qinglong/GPT-NER-multi-step-v2/openai_access/low_resource_data/scale_0604/ori_data")
    parser.add_argument("--train_type", type=str, help="directory for the example", default="train_all")
    parser.add_argument("--test_type", type=str,  help="directory for the example",default="test_50")
    parser.add_argument("--test_prefix", type=str,  help="file name for the example",default="test.50")
    parser.add_argument("--train_prefix", type=str, help="numebr for examples",default="new_train.json")
    parser.add_argument("--knn_example_file", type=str, help="numebr for examples",default="/data/qinglong/GPT-NER-multi-step-v2/openai_access/low_resource_data/scale_0604/ori_data/test_50.simcse.train_all.5.knn.json")
    parser.add_argument("--result_path", type=str, help="unfinished file")

    return parser

# Read a specific type of data file, process each line of data, and return it.
def read_type_data(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = f.readlines()
    data = [list(set(list(ast.literal_eval(d)))) for d in data]
    return data

# Read MRC data, concatenate the file name based on the given directory and prefix, read the file content and parse it into JSON format.
def read_mrc_data(dir_, prefix):
    file_name = os.path.join(dir_, f"{prefix}")
    with open(file_name, "r") as f:
        data = [json.loads(line.strip("\n")) for line in f.readlines()]
    return data

# Read the result file and return all lines in the file.
def read_results(dir_):
    file = open(dir_, "r")
    resulst = file.readlines()
    file.close()
    return resulst

# Read example data, with the default prefix "dev", and read the JSON file from the specified directory.
def read_examples(dir_, prefix="dev"):
    print("reading ...")
    file_name = os.path.join(dir_, f"mrc-ner.{prefix}")
    return json.load(open(file_name, encoding="utf-8"))

# Read index data.
def read_idx(file_name):
    print("reading ...")
    # file_name = os.path.join(dir_, f"{prefix}.knn.json")
    example_idx = []
    file = open(file_name, "r")
    for line in file:
        example_idx.append(json.loads(line.strip()))
    file.close()
    return example_idx

# Convert MRC data to prompt information.
def mrc2prompt(mrc_data, types_data, data_name="CONLL", example_idx=None, train_mrc_data=None, example_num=16, last_results=None):
    print("mrc2prompt ...")

    def get_example(index):
        exampel_prompt = ""
        for idx_ in example_idx[index][:example_num]:
            context = train_mrc_data[idx_]["text"]
            context_list = [char for char in context.strip()]
            labels = ""
            entities = train_mrc_data[idx_]["label"]
            
            last_ = 0
            for span_idx in range(len(entities)):
                start_ = entities[span_idx][0]
                end_ = entities[span_idx][1] 

                if labels != "":
                    labels += ""
                if last_ == start_:
                    labels += special_symbol[entities[span_idx][2]][0] + "".join(context_list[start_:end_]) + special_symbol[entities[span_idx][2]][1]
                else:
                    labels += "".join(context_list[last_:start_]) + ""+special_symbol[entities[span_idx][2]][0] + "".join(context_list[start_:end_]) + special_symbol[entities[span_idx][2]][1]
                last_ = end_

            if labels != "" and last_ != len(context_list):
                labels += ""
            labels += "".join(context_list[last_:])

            exampel_prompt += f"\"Given text\":\"{context}\"\n{{\"Marking result\":\"{labels}\"}}\n"
            
            # exampel_prompt += f"{prompt_label_name} entities: {labels}\n"
            # exampel_prompt += f"The given sentence is as follows:{labels}.\n"
        return exampel_prompt
        
    results = []
    special_symbol = {"Scale":("<scale>", "</scale>"),  "Item":("<item>", "</item>"), "Concept":("<concept>", "</concept>")}
    for item_idx in tqdm(range(len(mrc_data))):

        if last_results is not None and last_results[item_idx].strip() != "FRIDAY-ERROR-ErrorType.unknown":
            continue
        label_map = {'Scale': 'Scale', 'Measurement item': 'Item', 'Measurement concept': 'Concept'}
        item_ = mrc_data[item_idx]
        context = item_["text"]
        prompt_base = ""
        origin_label = item_["label"]
        types = types_data[item_idx]
        if types == ["null"]:
            prompt_base += f"Given text: “{context}”\n"
            results.append(prompt_base)
            continue
        sub_types = {t:(special_symbol[label_map[t]], FULL_DATA[data_name][label_map[t]][1], FULL_DATA[data_name][label_map[t]][2]) for t in types}
        prompt_base = "You are an excellent linguist and named entity recognition expert. The task is to mark the corresponding entities of "+",".join([f"“{t}”" for t in types])+" in the given text. To help you understand, the detailed descriptions of these "+str(len(types))+" types of entities are given below:\n"
        for i, t in enumerate(types):
            prompt_base += f"{i+1}.\"{t}\":{sub_types[t][1]}\n"   ######## Replace with detailed description
        prompt_base +=f"Marking rules are as follows:\n"
        for j, t in enumerate(types):
            prompt_base += f"{j+1}. If the entity “{t}” exists, add the marking symbols "+sub_types[t][0][0]+" and "+sub_types[t][0][1]+" to the left and right of each entity respectively.\n"
        prompt_base += f"{1+len(types)}. If there are no "+",".join([f"“{t}”" for t in types])+" in the given text, output the original text directly.\nOutput the results in the following JSON format: {\"Marking result\":\"Marked sentence\"}. Please note: Only output the results, do not output other content."        
        if example_num > 0:
            prompt_base+=f"Here are {example_num} examples:\n"
            prompt_base += get_example(index=item_idx)

        prompt_base += f"Given text: \"{context}\""


        # print(prompt_base)
        results.append(prompt_base)
    
    return results

# Perform named entity recognition access.
def ner_access(openai_access, ner_pairs, model, batch=10):
    print("tagging ...")
    results = []
    start_ = 0
    pbar = tqdm(total=len(ner_pairs))
    while start_ < len(ner_pairs):
        end_ = min(start_+batch, len(ner_pairs))
        if ner_pairs[start_:end_][0].startswith("Given text: “"):
            given_sentence = ner_pairs[start_:end_][0].replace('\n', '')[8:-1]
            json_string = f'{{"Marking result":"{given_sentence}"}}'
            print("*"*50)
            print(json_string)
            print("*"*50)
            results = results + [json_string]
        else:
            print(ner_pairs[start_:end_][0])
            anserwer = openai_access.get_multiple_sample(ner_pairs[start_:end_], model)[0]
            print("*"*50)
            print(anserwer)
            print("*"*50)
            results = results + [anserwer]
        pbar.update(end_-start_)
        start_ = end_
    pbar.close()
    return results

# Write data to a file.
def write_file(labels, dir_, last_name):
    print("writing ...")
    file_name = os.path.join(dir_, last_name)
    file = open(file_name, "w")
    for line in labels:
        file.write(str(line).strip()+'\n')
    file.close()
    # json.dump(labels, open(file_name, "w"), ensure_ascii=False)

# Test function, performing a series of data reading, processing, accessing, and writing operations.
def test(args):
    openai_access = AccessBase(
        engine="gpt-3.5-turbo-instruct",
        do_sample = False,
        temperature=0.0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1
    )
    zhipu_access = AccessBase(
        engine="glm-4-0520",
        # engine="glm-4",
        # engine="glm-3-turbo",
        do_sample = False,
        temperature=0.02,
        max_tokens=2048,
        top_p=0.7,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1
    )
    kimi_access = AccessBase(
        engine="moonshot-v1-8k",
        do_sample = False,
        temperature=0.0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1
    )
    # strategy = sys.argv[1]
    # bigmodel = sys.argv[2]
    # example_num = int(sys.argv[3])
    # data_path=sys.argv[4]
    # train_type = sys.argv[5]    
    # test_prefix = sys.argv[6]
    # train_prefix = sys.argv[7]
    # knn_word_sentence_num = int(sys.argv[8])

    access = zhipu_access
    ner_test = read_mrc_data(args.data_path, prefix=args.test_prefix)
    mrc_train = read_mrc_data(args.data_path, prefix=args.train_prefix)
    example_idx = read_idx(args.knn_example_file)
    types_data = read_type_data(os.path.join(args.result_path, f"tmp.zhipu.type.{args.test_prefix}"))

    prompts = mrc2prompt(mrc_data=ner_test, types_data=types_data,data_name="SCALE", example_idx=example_idx, train_mrc_data=mrc_train, example_num=args.example_num)
    results = ner_access(openai_access=access, ner_pairs=prompts, model=args.bigmodel, batch=1)
    # print(results)
    write_file(results, args.result_path, f"tmp.zhipu.{args.test_type}.{args.train_type}.{args.example_num}")

if __name__ == '__main__':
    # Get the command line argument parser.
    parser = get_parser()
    # Parse the command line arguments.
    args = parser.parse_args()
    # Call the test function.
    test(args)