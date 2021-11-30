import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
import math
import json

def clear(datadict):
    datalist = datadict["sample"]
    datalist = [d for d in datalist if isinstance(d, str)]
    datalist = [d for d in datalist if d != "\n"]
    datalist = [d.replace("\n", "") for d in datalist]
    datadict["sample"] = datalist

    return datadict

def load_json(filename):
    filepath = r"/remote-home/gzhao/retrival-engine/sales/data/corpus1.0/" + filename + "_corpus1.0.json"
    filejson = open(filepath, "r", encoding="utf-8")
    loadobject = json.load(filejson)
    return loadobject

def save_json(object, filename):
    filepath = r"/remote-home/gzhao/retrival-engine/sales/data/corpus1.0/" + filename + "_corpus1.0.json"
    filejson = open(filepath, "w", encoding="utf-8")
    json.dump(object, filejson, ensure_ascii=False)
    print("{} is saved correctly".format(filename))

def write_data_in_txt(data, filename, intent2id):
    filepath = r"/remote-home/gzhao/retrival-engine/sales/data/corpus1.0/" + filename + ".txt"
    with open(filepath, "w", encoding="utf-8") as f:
        for d in data:
            for i in range(len(d["sample"])):
                f.write(str(intent2id[d["intent"]]) + "\t" + d["sample"][i] + "\n")

def write_intent_in_txt(intent_list, filename):
    filepath = r"/remote-home/gzhao/retrival-engine/sales/data/corpus1.0/" + filename + ".txt"
    with open(filepath, "w", encoding="utf-8") as f:
        for intent in intent_list:
            f.write(intent + "\n")

def read_csv(datapath=r"/remote-home/gzhao/retrival-engine/sales/data/corpus1.0/corpus1.0.xlsx"):
    df = pd.read_excel(datapath)
    print("Demo of df is \n {}".format(df.head()))
    dflen = len(df)
    attrs = df.columns.tolist()
    print("Attributes of df is {}".format(attrs))

    data = []
    newdata = {}
    for i in tqdm(range(dflen)):
        if isinstance(df.loc[i, "FAQ标准问"], str):
            data.append(copy.deepcopy(newdata))
            newdata["intent"] = df.loc[i, "FAQ标准问"]
            newdata["sample"] = [df.loc[i, "相似问"]]
            newdata["answer"] = df.loc[i, "FAQ回答"]
        else:
            newdata["sample"].append(df.loc[i, "相似问"])
    data.append(newdata)
    data.pop(0)
    data = map(clear, data)
    data = list(data)

    # print(data[42])
    intent_list = [d["intent"] for d in data]
    answer_list = [d["answer"] for d in data]

    # intent_list = [d for d in intent_list if d != "\n"]
    # intent_list = [d.replace("\n", "") for d in intent_list]
    # for i, d in enumerate(answer_list):
    #     print("{}, {}".format(i, d))
    # answer_list = [d for d in answer_list if isinstance(d, str)]
    # answer_list = [d.replace("\n", "") for d in answer_list]

    # for i, d in enumerate(data):
    #     d["intent"]

    # for i, d in enumerate(answer_list):
    #     print("{}, {}".format(i, d))

    assert len(intent_list) == len(answer_list) == len(data)

    intent2id = {intent_list[i]: i for i in range(len(intent_list))}
    id2intent = {value: key for key, value in intent2id.items()}

    answer2id = {answer_list[i]: i for i in range(len(answer_list))}
    id2answer = {value: key for key, value in answer2id.items()}

    intent2answer = {intent_list[i]: answer_list[i] for i in range(len(intent_list))}
    answer2intent = {answer_list[i]: intent_list[i] for i in range(len(intent_list))}

    save_json(intent_list, "intent_list")
    save_json(answer_list, "answer_list")
    save_json(intent2id, "intent2id")
    save_json(id2intent, "id2intent")
    save_json(answer2id, "answer2id")
    save_json(id2answer, "id2answer")
    save_json(intent2answer, "intent2answer")
    save_json(answer2intent, "answer2intent")

    write_data_in_txt(data, "data", intent2id)
    write_intent_in_txt(intent_list, "intent")
    write_intent_in_txt(answer_list, "answer")

    print("Length of data is {}".format(len(data)))
    print("Intent Length is {}".format(len(intent_list)))
    print("All intents are {}".format(intent_list))
    print("Answer Length is {}".format(len(answer_list)))
    print("All answers are {}".format(answer_list))
    print("data[0] is {}".format(data[49]))
    print("data[1] is {}".format(data[1]))
    return data



if __name__ == "__main__":
    data = read_csv()
