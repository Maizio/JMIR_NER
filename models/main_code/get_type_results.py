import os
from tqdm import tqdm
from base_access import AccessBase
from logger import get_logger
import json
import argparse
from dataset_name import FULL_DATA
import random
import sys
import argparse
from simcse import SimCSE
import faiss
import numpy as np

# Set the random seed for reproducibility
random.seed(1)
# Get the logger for this module
logger = get_logger(__name__)

# os.environ["http_proxy"] = "http://localhost:10808"
# os.environ["https_proxy"] = "http://localhost:10809"

# Function to create an argument parser
def get_parser():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add command-line arguments with their descriptions and default values
    # parser.add_argument("--strategy", type=str, help="directory for the input")
    parser.add_argument("--bigmodel", type=str, help="file name for the input", default="zhipu")
    parser.add_argument("--knn_num", type=int, help="file name for the training set", default=20)
    parser.add_argument("--model_path", type=str, help="file name for the example", default="/data/qinglong/GPT-NER-test/models/acge_text_embedding")
    parser.add_argument("--data_path", type=str, help="dataset name for the input", default="/data/qinglong/GPT-NER-multi-step-v2/openai_access/low_resource_data/scale_0620/type_random_100")
    parser.add_argument("--train_type", type=str, help="directory for the example", default="train_all")
    parser.add_argument("--test_type", type=str, help="directory for the exampl", default="test_100")
    parser.add_argument("--test_prefix", type=str, help="file name for the example", default="test.100")
    # parser.add_argument("--train_prefix", type=str, help="numebr for examples")
    parser.add_argument("--result_path", type=str, help="unfinished file", default="/data/qinglong/GPT-NER-multi-step-v2/openai_access/low_resource_data/scale_0620/type_random_100")

    return parser

# Function to read data from a file
def read_data(dir_, prefix="test"):
    # Construct the full file name
    file_name = os.path.join(dir_, f"{prefix}")
    # Open the file in read mode
    with open(file_name, "r") as f:
        # Read each line, strip the newline character, parse as JSON, and store in a list
        data = [json.loads(line.strip("\n")) for line in f.readlines()]
    return data

# Function to write data to a file
def write_file(labels, dir_, last_name):
    print("writing ...")
    # Construct the full file name
    file_name = os.path.join(dir_, last_name)
    # Open the file in write mode
    file = open(file_name, "w")
    # Write each line as JSON string to the file
    for line in labels:
        file.write(json.dumps(line, ensure_ascii=False)+'\n')
    # Close the file
    file.close()

# Function to convert MRC data to prompts
def mrc2prompt(mrc_data, train_mrc_data, example_num, example_idx, data_name="CONLL",  last_results=None):
    print("mrc2prompt ...")
    # List to store the generated prompts
    results = []

    # Inner function to get example prompts
    def get_example(index):
        # Initialize an empty example prompt string
        exampel_prompt = ""
        # Mapping of label names
        label_map = {"Scale": "量表", "Item": "测量项目", "Concept": "测量概念"}
        # Iterate over the example indices
        for idx_ in example_idx[index][:example_num]:
            # Get the text context from the training data
            context = train_mrc_data[idx_]["text"]
            # Get the labels from the training data
            labels = train_mrc_data[idx_]["label"]
            # Extract unique types from the labels using the label map
            types = set([label_map[entity[2]] for entity in labels])
            # If no types are found, set it to "null"
            if types == set():
                types = {"null"}

            # Construct the example prompt
            exampel_prompt += f"\"给定句子\":\"{context}\"\n\"输出\":\"{types}\"\n"

        return exampel_prompt

    # Iterate over the MRC data with a progress bar
    for item_idx in tqdm(range(len(mrc_data))):
        # If there are last results and the current item is not an error, skip it
        if last_results is not None and last_results[item_idx].strip() != "FRIDAY-ERROR-ErrorType.unknown":
            continue

        # Get the current item from the MRC data
        item_ = mrc_data[item_idx]
        # Get the text context from the item
        context = item_["text"]
        # origin_label = item_["label"]
        # Initialize an empty prompt string
        prompt = ""
        # Get the label types from the full data
        label_types = FULL_DATA[data_name]
        # Base prompt for the task
        # prompt_base = "你是一个优秀的语言学家和命名实体识别专家。你的任务是从给定文本和给定实体类型列表{\"量表\",\"测量概念\",\"测量项目\"},列出文本中可能存在的实体类型。"+f"为帮助你理解,下面给出{len(label_types.keys())}类实体解释及可能取值说明:\n"
        prompt_base = "你是一个优秀的语言学家和命名实体识别专家。你的任务是根据给定的文本和实体类型列表{\"量表\",\"测量概念\",\"测量项目\"}，识别并列出文本中存在的实体类型。以下是实体类型及其特征的详细说明：\n"
        prompt += prompt_base
        # Add the description of each label type to the prompt
        for i, label_type in enumerate(label_types.values()):
            prompt += f"{i}.“{label_type[0]}”:{label_type[1]}\n" 

        # Add instructions to the prompt
        prompt += f"按如下json格式进行结果输出:{{实体类型1,实体类型2}}。请注意:\n1.只需要输出结果,不要输出其他内容。\n2.所有输出的实体类型必须在实体类型列表中,如果句子中不存在任何实体类型,则输出{{\"null\"}}。\n3.不要重复输出结果列表中已经存在的实体类型。\n"
        # prompt += f"The given sentence: {context}\nThe labeled sentence:"
        # If there are examples, add them to the prompt
        if example_num > 0:
            prompt += f"下面是{example_num}个例子:\n"
            prompt += get_example(item_idx)
        # Add the current text context to the prompt
        prompt += f"给定文本如下:“{context}”"
        # print(prompt)

        # Append the generated prompt to the results list
        results.append(prompt)

    return results

# Function to perform NER access
def ner_access(openai_access, ner_pairs, model, batch=10):
    print("tagging ...")
    # List to store the results
    results = []
    # Starting index
    start_ = 0
    # Create a progress bar
    pbar = tqdm(total=len(ner_pairs))
    # Loop until all pairs are processed
    while start_ < len(ner_pairs):
        # Ending index for the current batch
        end_ = min(start_ + batch, len(ner_pairs))
        # Print the first pair in the current batch
        print(ner_pairs[start_:end_][0])
        # Create a message for the API call
        message = [{"role": "user", "content": ner_pairs[start_:end_][0]}]
        # Get the answer from the API
        answer = openai_access.get_multiple_sample(message, model)
        print("*" * 50)
        print(answer[0])
        print("*" * 50)
        # Extend the results list with the answers
        results = results + answer
        # Update the progress bar
        pbar.update(end_ - start_)
        # Move to the next batch
        start_ = end_
    # Close the progress bar
    pbar.close()
    return results

# Function to write the result file
def write_result_file(labels, dir_, last_name):
    print("writing ...")
    # Construct the full file name
    file_name = os.path.join(dir_, last_name)
    # Open the file in write mode
    file = open(file_name, "w")
    # Write each line to the file
    for line in labels:
        file.write(line.strip() + '\n')
    # Close the file
    file.close()
    # json.dump(labels, open(file_name, "w"), ensure_ascii=False)

# Function to compute SimCSE KNN
def compute_simcse_knn(test_mrc_data, train_mrc_data, model_path, knn_num, test_index=None):  ##### Find n data with high text similarity from the training dataset for each data in the test dataset
    # Initialize the SimCSE model
    sim_model = SimCSE(model_path)

    # Extract the text from the training data
    train_sentence = [item["text"] for item in train_mrc_data]
    # Create an index for the training sentences
    train_sentence_index = [value for value, key in enumerate(train_sentence)]

    # Encode the training sentences
    embeddings = sim_model.encode(train_sentence, batch_size=128, normalize_to_unit=True, return_numpy=True)
    # Initialize the quantizer
    quantizer = faiss.IndexFlatIP(embeddings.shape[1])
    # quantizer = faiss.GpuIndexFlatL2(embeddings.shape[1])
    index = quantizer
    # Add the embeddings to the index
    index.add(embeddings.astype(np.float32))
    # Set the number of probes (default in simcse is 10)
    index.nprobe = min(10, len(train_sentence))
    # Move the index to CPU
    index = faiss.index_gpu_to_cpu(index)

    train_index = index

    # Lists to store the example indices and values
    example_idx = []
    example_value = []

    # If no test index is provided, process all test data
    if test_index is None:
        # Iterate over the test data
        for idx_ in range(len(test_mrc_data)):
            # Get the text context from the test data
            context = test_mrc_data[idx_]["text"]

            # Encode the context
            embedding = sim_model.encode([context], batch_size=128, normalize_to_unit=True, keepdim=True, return_numpy=True)
            # Search for the nearest neighbors
            top_value, top_index = train_index.search(embedding.astype(np.float32), knn_num)

            # Convert the indices to the original sentence indices and append to the list
            example_idx.append([train_sentence_index[int(i)] for i in top_index[0]])
            # Convert the values to float and append to the list
            example_value.append([float(value) for value in top_value[0]])

        return example_idx, example_value

    # If a test index is provided, process only the specified indices
    for idx_, sub_index in enumerate(test_index):
        # Skip if the sub_index is not 0
        if sub_index != 0:
            continue
        # Get the text context from the test data
        context = test_mrc_data[idx_]["text"]

        # Encode the context
        embedding = sim_model.encode([context], batch_size=128, normalize_to_unit=True, keepdim=True, return_numpy=True)
        # Search for the nearest neighbors
        top_value, top_index = train_sentence_index.search(embedding.astype(np.float32), knn_num)

        # Convert the indices to the original sentence indices and append to the list
        example_idx.append([train_sentence_index[int(i)] for i in top_index[0]])
        # Convert the values to float and append to the list
        example_value.append([float(value) for value in top_value[0]])

    return example_idx, example_value

# Function to perform the test
def test(args, train_mrc_data, index_):
    # Initialize the OpenAI access object
    openai_access = AccessBase(
        engine="gpt-3.5-turbo-instruct",
        do_sample=False,
        temperature=0.0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1
    )
    # Initialize the ZhiPu access object
    zhipu_access = AccessBase(
        # engine="glm-4",
        # engine="glm-3-turbo",
        engine="glm-4-0520",
        do_sample=False,
        temperature=0.02,
        max_tokens=500,
        top_p=0.7,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1
    )
    # Initialize the Kimi access object
    kimi_access = AccessBase(
        engine="moonshot-v1-8k",
        do_sample=False,
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
    # data_path = sys.argv[4]
    # train_type = sys.argv[5]
    # test_prefix = sys.argv[6]
    # train_prefix = sys.argv[7]
    # knn_word_sentence_num = int(sys.argv[8])

    # Set the access object to ZhiPu
    access = zhipu_access
    # Read the test data
    ner_test = read_data(args.data_path, prefix=args.test_prefix)

    # Convert the MRC data to prompts
    prompts = mrc2prompt(mrc_data=ner_test, train_mrc_data=train_mrc_data, example_num=args.knn_num, example_idx=index_, data_name="SCALE")
    # Perform NER access
    results = ner_access(openai_access=access, ner_pairs=prompts, model=args.bigmodel, batch=1)
    # print(results)
    # Write the results to a file
    write_result_file(results, args.result_path, f"tmp.zhipu.type.{args.test_prefix}.{args.knn_num}")

if __name__ == '__main__':
    # Get the argument parser
    parser = get_parser()
    # Parse the command-line arguments
    args = parser.parse_args()
    # Read the test data
    test_mrc_data = read_data(args.data_path, prefix=args.test_prefix)
    # Read the training data
    train_mrc_data = read_data(args.data_path, prefix="train_dev.json")
    # If the KNN number is 0, perform the test without computing KNN
    if args.knn_num == 0:
        test(args, train_mrc_data, None)
        sys.exit(0)
    else:
        # Compute the SimCSE KNN
        index_, values = compute_simcse_knn(test_mrc_data, train_mrc_data, args.model_path, args.knn_num, test_index=None)
        # Construct the example file name
        example_file = f"type.{args.test_type}.simcse.{args.train_type}.{args.knn_num}.knn.json"
        # example_sentence_file = f"{args.test_type}.simcse.{args.train_type}.{args.knn_num}.knn.sentence.json"
        # Write the example indices to a file
        write_file(index_, dir_=args.result_path, last_name=example_file)
        # Perform the test with the computed KNN indices
        test(args, train_mrc_data, index_)
