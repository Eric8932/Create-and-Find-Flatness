import os
import json
import pandas as pd

dataset = {
    'ag'     : None,
    'yelp'   : None,
    'dbpedia': None,
    'amazon' : None,
    'yahoo'  : None,
}

DATA_DIR = "../data/data"
SAVE_DIR = "../data"

def preprocess(dataset_name, mode):

    file_name = dataset_name + "_to_squad-" + mode + "-v2.0.json"

    folder = os.path.join(SAVE_DIR, dataset_name)
    if not os.path.exists(folder):
        os.mkdir(folder)
    file_path = os.path.join(folder, mode + ".tsv")

    file_write = open(file_path,'w',encoding = 'utf-8')
    file_write.write('text_a\tlabel\n')

    with open(os.path.join(DATA_DIR, file_name)) as json_file:
        data = json.load(json_file)
        data = data["data"]
        answers = []
        contexts = []
        questions = []
        for i in range(len(data)):
            paragraph = data[i]["paragraphs"][0]
            context = paragraph["context"]
            qas = paragraph["qas"][0]
            question = qas["question"]
            answer = qas["answers"][0]["text"]
            contexts.append(context)
            questions.append(question)
            answers.append(answer)
        if dataset_name != 'amazon':
            answers_id = cls_name2id(answers, dataset_name)
        else:
            # reuse the label id from yelp to strictly align
            answers_id = cls_name2id(answers, 'yelp')

    for i in range(len(contexts)):
        label = answers_id[i]
        if len(label)==0 or label==' ' or len(label)>2:
            continue
        file_write.write(contexts[i].replace('\t',' '))
        file_write.write('\t')
        file_write.write(label)
        file_write.write('\n')
    file_write.close()


def cls_name2id(classes, dataset_name):
    name2id = dataset[dataset_name]
    if name2id is None:
        name2id = {}
        cls_id = 0
        ids = []
        for cls in classes:
            if cls not in name2id:
                name2id[cls] = cls_id
                cls_id += 1
            ids.append(str(name2id[cls]))
        dataset[dataset_name] = name2id
    else:
        ids = [str(name2id[cls]) for cls in classes]
    print(dataset[dataset_name])
    return ids


if __name__ == "__main__":
    datasets = ['ag', 'yelp', 'amazon', 'dbpedia', 'yahoo']
    modes = ['train', 'test']
    for ds in datasets:
        for mode in modes:
            preprocess(ds, mode)


