# es插入数据，检索数据
import copy

from elasticsearch import Elasticsearch
import time
import json
from tqdm import tqdm #进度条
from elasticsearch import helpers
from utils import load_test, load_id2intent


#####################创建索引####################################

es = Elasticsearch()

def deleteInices(my_index):
    if True and es.indices.exists(my_index):  #确认删除再改为True
        print("删除之前存在的")
        es.indices.delete(index=my_index)
    
def createIndex(my_index, my_doc):
    # index settings
    settings = \
    {
        "mappings": {
                my_doc :{
                "properties": {
                    "my_id": {"type": "integer"},
                    "my_word": {"type": "text",
                                "analyzer": "ik_smart",
                                "search_analyzer": "ik_smart"}
                }
            }
        }
    }
    # create index
    es.indices.create(index=my_index,
                      ignore=400,
                      mappings=settings["mappings"])
    print("创建index成功！")

def mainCreateIndex():
    # 调用后创建index
    my_index = "word2vec_index"
    my_doc = "my_doc"
    deleteInices(my_index)
    createIndex(my_index, my_doc)


#####################插入数据####################################

def getAllWords(path=r"/remote-home/gzhao/retrival-engine/sales/data/corpus1.0/intent.txt"):
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for i, item in enumerate(f.readlines()):
            words.append((i,item.strip()))
    return words

def insertData(words, my_index, my_doc, one_bulk):
    #插入数据
    #one_bulk表示一个bulk里装多少个
    body = []
    body_count = 0  #记录body里面有多少个.

    print("共需要插入%d条..."%len(words))
    pbar = tqdm(total=len(words))

    for id, word in words:
        data1 = { "my_id": id,
                  "my_word": word}
        every_body = \
        {
            "_index": my_index,
            "_type": my_doc,
            "_source": data1
        }

        if body_count<one_bulk:
            body.append(every_body)
            body_count+=1
        else:
            helpers.bulk(es, body) #还是要用bulk啊，不然太慢了
            pbar.update(one_bulk)
            body_count = 0
            body = []
            body.append(every_body)
            body_count+=1

    if len(body)>0:
        #如果body里面还有，则再插入一次（最后非整块的）
        helpers.bulk(es, body)
        # pbar.update(len(body))
        print('done2')

    pbar.close()
    print("插入数据完成!")

def mainInsert():
    # 调用后插入数据
    my_index = "word2vec_index"
    my_doc = "my_doc"
    words = getAllWords()
    insertData(words, my_index, my_doc, one_bulk=5000)


#####################检索数据####################################

def keywordSearch(keywords1, my_index, my_doc):
    #根据 keywords1 来查找，倒排索引
    my_search1 = \
        {
            "query" : {
                "match" : {
                    "my_word" : keywords1
                }
            }
        }

    # reord start time
    start_time = time.time()

    search_result = es.search(index=my_index,
                              body=my_search1,
                              scroll='10m',
                              size=12)

    search_id = [d["_source"]["my_id"] for d in search_result["hits"]["hits"]]
    search_score = [d["_score"] for d in search_result["hits"]["hits"]]
    search_intent = [d["_source"]["my_word"] for d in search_result["hits"]["hits"]]
    search_output = [(search_id[i], search_intent[i], search_score[i]) for i in range(len(search_id))]

    scroll_id = search_result["_scroll_id"]
    es.clear_scroll(scroll_id = scroll_id)

    # print(search_result)
    # print(search_result)
    # print(search_id)
    # print(search_score)
    # print(search_intent)
    # print(len(search_result['hits']))

    print("用户query为 {}".format(keywords1))
    print("粗排匹配到%d条意图点" % len(search_output))
    print(search_output)

    # record end time
    end_time = time.time()
    print("共耗时{}s".format(end_time-start_time))

    return search_output, end_time - start_time

# mainSearch 用来做单个测试样例的调试
def mainSearch():
    # 调用后检索数据
    my_index = "word2vec_index"
    my_doc = "my_doc"
    keywords1 = "请问办理信用卡要多少钱吗"
    keywordSearch(keywords1, my_index, my_doc)


def testSearch():
    my_index = "word2vec_index"
    my_doc = "my_doc"

    testdata = load_test()
    id2intent = load_id2intent()

    right = 0
    total = 0
    total_time = 0

    intent_es_search_hit = {id :0 for id in id2intent.keys()}
    intent_es_search_total = copy.deepcopy(intent_es_search_hit)
    wrong_list = {id :[] for id in id2intent.keys()}

    for i, data in enumerate(testdata):
        label = int(data[0])
        demo = data[1]
        result, t = keywordSearch(demo, my_index, my_doc)
        print("正确意图点是 {}\n".format(id2intent[str(label)]))
        re_label = [item[0] for item in result]
        intent_es_search_total[str(label)] += 1

        if label in re_label:
            right += 1
            intent_es_search_hit[str(label)] += 1
        else:
            wrong_list[str(label)].append(demo)

        total += len(result)
        total_time += t

    acc = right / len(testdata)
    print("粗排精度为{}".format(acc))
    print("平均搜索样本数量为{}".format(total/len(testdata)))
    print("总耗时{}s,平均耗时{}s".format(total_time,total_time/len(testdata)))

    for key, value in intent_es_search_total.items():
        if value == 0:
            intent_es_search_total[key] = 1

    intent_es_search_recall = {id: intent_es_search_hit[id] / intent_es_search_total[id]
                               for id, _ in intent_es_search_hit.items()}

    intent_es_search_recall = {id2intent[str(key)]: value for key, value in intent_es_search_recall.items()}
    wrong_list = {id2intent[str(key)]:value for key, value in wrong_list.items()}

    print("Search Hits ...")
    for id, hit in intent_es_search_hit.items():
        print(id2intent[str(id)], hit)

    print("Search total ...")
    for id, total in intent_es_search_total.items():
        print(id2intent[str(id)], total)

    print("Search recall ...")
    for intent, recall in intent_es_search_recall.items():
        print(intent, recall)
        for ii in wrong_list[intent]:
            print(ii)
        print("\n")

if __name__ == "__main__":
    # mainCreateIndex()
    # mainInsert()
    # mainSearch()
    testSearch()