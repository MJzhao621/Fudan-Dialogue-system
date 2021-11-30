import json

def load_test(path=r"/remote-home/gzhao/retrival-engine/sales/data/corpus1.0/data.txt"):
    test = open(path, encoding='utf-8')
    lines = test.readlines()
    test_data = []
    for line in lines:
        print(line)
        d = line.split('\t')
        test_data.append((d[0], d[1][:-1]))
    print("Loading test_file({} samples)".format(len(test_data)))
    return test_data

def load_id2intent(path=r"/remote-home/gzhao/retrival-engine/sales/data/corpus1.0/id2intent_corpus1.0.json"):
    fjson = open(path, "r", encoding="utf-8")
    data = json.load(fjson)
    return data
