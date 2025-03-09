import os
from tqdm import tqdm
from base_access import AccessBase
from logger import get_logger
import json
import argparse
from dataset_name import FULL_DATA
import random
from simcse import SimCSE
import numpy as np
import faiss
import sys
import re
import ast

# Set proxy environment variables
os.environ["http_proxy"] = "http://localhost:10808"
os.environ["https_proxy"] = "http://localhost:10809"
# Set random seed for reproducibility
random.seed(1)
# Get a logger instance for the current module
logger = get_logger(__name__)

# Function to create and return an argument parser
def get_parser():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    parser.add_argument("--mrc-dir", type=str, default="/data/qinglong/GPT-NER-multi-type/openai_access/low_resource_data/scale_0604/ori_data", help="directory for the mrc input")
    parser.add_argument("--mrc-name", type=str, default="test.50",help="file name for the mrc input")
    parser.add_argument("--gpt-dir", type=str, default="/data/qinglong/GPT-NER-multi-type/openai_access/low_resource_data/scale_0604/ori_data",help="directory for the gpt input")
    parser.add_argument("--gpt-name", type=str, default="tmp.zhipu.test_50.train_all.5",help="file name  for the gpt input")
    parser.add_argument("--data-name", type=str,default="SCALE",  help="dataset name for the input")
    parser.add_argument("--knn-file", default="/data/qinglong/GPT-NER-multi-type/openai_access/low_resource_data/scale_0604/ori_data/verify_test_50.simcse.train_all.5.knn.json", type=str, help="knn file for the input")
    # parser.add_argument("--knn-file", default="None", type=str, help="knn file for the input")
    parser.add_argument("--write-dir", type=str, default="/data/qinglong/GPT-NER-multi-type/openai_access/low_resource_data/scale_0604/ori_data", help="directory for the output")
    parser.add_argument("--write-name", type=str, default="zhipu.test_50.simcse.train_all.5",help="file name for the output")
    parser.add_argument("--knn-num", type=int, default=5, help="numebr for the knn")
    
    return parser

# Function to read MRC data from a file
def read_mrc_data(dir_, prefix="test"):

    file_name = os.path.join(dir_, f"{prefix}")
    with open(file_name, "r") as f:
        data = [json.loads(line.strip("\n")) for line in f.readlines()]
    return data

# Function to read results from a file
def read_results(dir_, prefix="test"):
    file_name = os.path.join(dir_, prefix)
    print(f"read ... {file_name}")

    file = open(file_name, "r", encoding="utf-8")
    results = []
    for line in tqdm(file):    
        results.append(ast.literal_eval(line.strip()))
    file.close()
    return results

# Function to read KNN file
def read_knn_file(file_name):
    file = open(file_name, "r")
    results = []
    for line in tqdm(file):
        results.append(json.loads(line.strip()))
    file.close()
    return results

# Function to transfer prompt
def transferPrompt(mrc_data, gpt_results, data_name="CONLL", knn_results=None, knn_num=14):
    print("tansferring prompt ...")

    # Function to get words from a labeled sentence
    def get_words(labeled_sentence):
        # Define symbols for different entity types
        symbol = {"Scale":("<scale>", "</scale>"),  "Item":("<item>", "</item>"), "Concept":("<concept>", "</concept>")}
        word_list = []
        # Iterate through different entity types
        for type in ["Scale", "Item", "Concept"]:
            entity_symbol = symbol[type]
            symbol_start = entity_symbol[0]
            symbol_end = entity_symbol[1]
            # Create a regular expression pattern to match the entity tags
            pattern = re.escape(symbol_start) + '(.*?)' + re.escape(symbol_end)
            # Find all matches of the pattern in the labeled sentence
            matches= list(re.finditer(pattern,labeled_sentence))
            for i, match in enumerate(matches):
                # Get the start and end positions of the match
                start,end = match.span()
                # Append the matched word, start and end positions, and entity type to the word list
                word_list.append((match.group(1),start, end, type))
        # Sort the word list by the start position
        word_list = sorted(word_list, key=lambda x:x[1])
        return word_list
    
    # Function to get KNN prompt
    def get_knn(index_, test_label):
        # If the KNN results for the index are empty, return None
        if len(knn_results[index_]) == 0:
            return None
        prompt = ""
        # Iterate through the first knn_num KNN results
        for sentence, word, label_type, _ in knn_results[index_][:knn_num]:
            # Get the transferred label
            transfered_label = FULL_DATA[data_name][label_type][0]
            # Determine the answer based on whether the transferred label matches the test label
            answer = "是" if transfered_label == test_label  else "否"

            # Construct the prompt with the sentence, word, test label, and answer
            prompt += f"“{word}”在给定的句子：“{sentence}”中是一个{test_label}实体吗? 请用是或否来回答。\n{answer}\n"
        
        return prompt

    prompts = []
    entity_index = []
    prompts_nums = []
    knn_idx = 0
    # Iterate through the GPT results with a progress bar
    for item_idx in tqdm(range(len(gpt_results))):
        item_ = gpt_results[item_idx]
        context = item_["输出"]
        ori_text = mrc_data[item_idx]["text"]
        # Get the entity list from the context
        entity_list = get_words(context.strip())
        prompts_num = 0
        # Iterate through the entity list
        for key, entity in enumerate(entity_list):
            # Get the transferred label
            transfered_label = FULL_DATA[data_name][entity[3]][0]
            sub_prompt = FULL_DATA[data_name][entity[3]][1]
            sub_prompt_detail = FULL_DATA[data_name][entity[3]][2]
            # Construct the prompt with the transferred label and detailed description
            prompt = f"你是一个优秀的语言学家和命名实体识别专家。任务是验证给定句子中提取的单词是否为“{transfered_label}”类实体，其详细描述：“{sub_prompt_detail}”。\n"

            if knn_results is None:
                # Construct the prompt without KNN examples
                prompt += f"请问“{entity[0]}”在给定的句子:“{ori_text}”中是一个{transfered_label}实体吗? 请用是或否来回答。注意:只需要输出答案，不要输出其他内容。\n"
                prompts.append(prompt)
                entity_index.append(((item_idx, key, entity[3].lower())))
                prompts_num += 1
            else:
                # Add a line indicating the following are examples
                prompt+="下面是一些例子:\n"
                knn_prompt = get_knn(knn_idx, test_label=transfered_label)
                if knn_prompt != "":
                    knn_idx += 1
                    # Add the KNN prompt to the main prompt
                    prompt += knn_prompt
                    # Add the question to the main prompt
                    prompt += f"请问“{entity[0]}”在给定的句子:“{ori_text}”中是一个{transfered_label}实体吗? 请用是或否来回答。注意:只需要输出答案，不要输出其他内容。\n"
                    prompts.append(prompt)
                    entity_index.append((item_idx, key, entity[3].lower()))
                    prompts_num += 1
        prompts_nums.append(prompts_num)
    return prompts, entity_index, prompts_nums

# Function to access NER service
def ner_access(openai_access, prompts,model, batch=16):
    print("accessing ...")
    results = []
    start_ = 0

    pbar = tqdm(total=len(prompts))
    while start_ < len(prompts):
        end_ = min(start_+batch, len(prompts))
        print(prompts[start_:end_][0])
        answer = openai_access.get_multiple_sample(prompts[start_:end_],model)
        results = results + answer
        print("*"*50)
        print(answer[0])
        print("*"*50)
        # Update the progress bar
        pbar.update(end_-start_)
        start_ = end_
    pbar.close()
    return results

# Function to construct final results
def construct_results(gpt_results,entity_index, prompts_num, verify_results):

    # Function to justify the answer
    def justify(string_):
        if string_ == "是":
            return "yes"
        if string_ == "否":
            return "no"
        return ""

    # Function to remove the nth tag from the text
    def remove_nth_tag(text, tag_name, n):
        # Regular expression pattern to match the specified tag and its content
        pattern = f"<{tag_name}>(.*?)</{tag_name}>"
        
        # Find all matches of the pattern in the text
        matches = list(re.finditer(pattern, text))
        
        # If the nth tag exists
        if len(matches) >= n:
            # Get the nth match
            match = matches[n-1]
            
            # Remove the nth match from the text
            start, end = match.span()  # Get the start and end positions of the match
            text = text[:start] + match.group(1) + text[end:]
        
        return text

    results = []
    start_ = 0
    # Iterate through the GPT results
    for idx_, item in enumerate(gpt_results):
        # Get the context from the item
        context = item["输出"]
        now_num = prompts_num[idx_]
        for sub_idx in range(now_num):
            num = start_ + sub_idx
            # Justify the verification result
            if justify(verify_results[num].strip()) == "yes":
                continue
            elif justify(verify_results[num].strip()) == "no":
                context = remove_nth_tag(context, entity_idx[num][2], sub_idx)
        start_ += now_num
        results.append({"输出":context})
    return results

# Function to write data to a file
def write_file(labels, dir_, last_name):
    print("writing ...")

    file_name = os.path.join(dir_, last_name)
    file = open(file_name, "w")
    for line in labels:
        file.write(json.dumps(line, ensure_ascii=False))
        file.write("\n")
    file.close()

if __name__ == '__main__':
    # test()

    # Get the argument parser
    parser = get_parser()
    # Parse the command line arguments
    args = parser.parse_args()

    # Create an AccessBase instance for GPT-3.5-turbo-instruct
    openai_access = AccessBase(
        engine="gpt-3.5-turbo-instruct",
        do_sample=False,
        temperature=0.0,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1
    )
    # Create an AccessBase instance for GLM-4
    zhipu_access = AccessBase(
        engine="glm-4",
        do_sample = False,
        temperature=0.02,
        max_tokens=500,
        top_p=0.7,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1
    )

    # Read MRC test data
    mrc_test = read_mrc_data(dir_=args.mrc_dir, prefix=args.mrc_name)
    # Read GPT results
    gpt_results = read_results(dir_=args.gpt_dir, prefix=args.gpt_name)

    knn_results = None
    if args.knn_file != "None":
        # Read KNN file if specified
        knn_results = read_knn_file(file_name=args.knn_file)

    # Transfer prompt
    prompts, entity_idx, prompts_nums = transferPrompt(mrc_data=mrc_test, gpt_results=gpt_results, data_name=args.data_name, knn_results=knn_results, knn_num=args.knn_num)
    print(sum(prompts_nums))
    # Access NER service
    verify_results = ner_access(openai_access=zhipu_access, prompts=prompts, model="zhipu" ,batch=1)
    # Write verification results to a file
    write_file(verify_results, args.mrc_dir, "verify_results" )
    # with open("/data/qinglong/GPT-NER-multi-type/openai_access/low_resource_data/scale_0604/ori_data/verify_results", "r", encoding="utf-8") as f:
    #     verify_results = f.readlines()
    # Construct final results
    final_results = construct_results(gpt_results=gpt_results, entity_index=entity_idx, prompts_num=prompts_nums, verify_results=verify_results)

    # print(final_results)

    write_file(labels=final_results, dir_=args.write_dir, last_name=args.write_name)
    # write_file(labels=verify_results, dir_=args.write_dir, last_name="only.verify.1")
