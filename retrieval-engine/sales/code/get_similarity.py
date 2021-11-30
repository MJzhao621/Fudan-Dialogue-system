import torch
from transformers import BertTokenizer,BertModel
import torch.nn.functional as F
# from sales.load_json import load_intent
from retrival_bert_server import Bi_Encoder, Cross_Encoder
from utils import load_test
import numpy as np
from search_word import keywordSearch
import json
import time

def load_intent(intentpath=r"/remote-home/gzhao/retrival-engine/sales/data/corpus1.0/intent_list_corpus1.0.json"):
    fjson = open(intentpath, "r", encoding="utf-8")
    intent = json.load(fjson)
    return intent

def load_pretrained_model(pretrained_path=r'D:\PycharmProject\Project1\retrival-engine\sales\chinese-bert',
                          deviice=torch.device(0)):
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    model = BertModel.from_pretrained(pretrained_path)
    return tokenizer, model

def load_finetune_model(finetune_path=r"/remote-home/gzhao/retrival-engine/sales/save/bert-finetune-crossencoder-02.pt",
                        pretrained_path=r'/remote-home/gzhao/retrival-engine/sales/chinese-bert',
                        device=torch.device(0)):
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    model = BertModel.from_pretrained(pretrained_path)
    similarity_model = Cross_Encoder(model=model, device=device).to(device)
    similarity_model.load_state_dict(torch.load(finetune_path))
    return tokenizer, similarity_model

def get_last_hidden(text, tokenizer, model):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
    # print("input_ids = {}".format(input_ids))
    # print("Length of input_ids = {}, and Length of text is {}. So the number of special tokens is {}"
    #       .format(input_ids.size(1), len(text), input_ids.size(1) - len(text)))
    outputs = model(input_ids)
    last_hidden = outputs[0]
    # print("Size of last hidden is {}".format(last_hidden.size()))
    return last_hidden

def get_sentence_embedding(last_hidden, method="first-dim"):
    # method = "first-dim" : employ the first vector of last hidden as the sentence representaton
    # method = "mean" : employ the mean vector of all dimensions as the sentence representation
    if method == "first-dim":
        return last_hidden[:,0]
    elif method == "mean":
        return torch.mean(last_hidden, axis=1)
    else:
        raise ValueError("Method cannot be found")

def score_query_and_intent_pretrained(query, intent, tokenizer, model):
    # query_hidden = get_last_hidden(query, tokenizer, model)
    # query_embedding = get_sentence_embedding(query_hidden)
    # intent_hidden = get_last_hidden(intent, tokenizer, model)
    # intent_embedding = get_sentence_embedding(intent_hidden)
    # score = torch.matmul(query_embedding, intent_embedding.T)
    return model.score(query, intent, tokenizer)

def trail(query, intent, output_type="finetune"):
    pretrained_tokenizer, pretrained_model = load_pretrained_model()
    finetune_tokenizer, finetune_model = load_finetune_model()
    finetune_model = finetune_model.bert_model
    score_pretrained = score_query_and_intent_pretrained(query, intent, pretrained_tokenizer, pretrained_model)
    score_finetune = score_query_and_intent_pretrained(query, intent, finetune_tokenizer, finetune_model)
    if output_type == "finetune":
        print("Finetune Model Output Score = {}".format(score_finetune))
    elif output_type == "pretrained":
        print("Pretrained Model Output Score = {}".format(score_pretrained))
    elif output_type == "both":
        print("Pretrained Model Output Score = {}\nFinetune Model Output Score = {}"
              .format(score_pretrained, score_finetune))
    else:
        raise ValueError("Output Type cannot be found")

def main(query):
    pretrained_tokenizer, pretrained_model = load_pretrained_model()
    finetune_tokenizer, finetune_model = load_finetune_model()
    intent_list = load_intent()
    for intent in intent_list:
        # score_pretrained = score_query_and_intent_pretrained(query, intent, pretrained_tokenizer, pretrained_model)
        score_finetune = score_query_and_intent_pretrained(query, intent, finetune_tokenizer, finetune_model)
        print("Intent = {}, Finetune = {}".format(intent, score_finetune))


def demo(user_text, intent):
    tokenizer, model = load_pretrained_model()
    query_last_hidden = get_last_hidden(user_text, tokenizer, model)
    query_sentence_embedding = get_sentence_emobedding(query_last_hidden, "mean")

    intent_last_hidden = get_last_hidden(intent, tokenizer, model)
    intent_sentence_embedding = get_sentence_emobedding(intent_last_hidden, "mean")

    score = torch.dot(query_sentence_embedding.squeeze(0), intent_sentence_embedding.squeeze(0))
    cosine_similarity = F.cosine_similarity(query_sentence_embedding.squeeze(0), intent_sentence_embedding.squeeze(0), dim=0)
    print("Intent = {}, Score = {}".format(intent, score))
    print("Intent = {}, Score = {}".format(intent, cosine_similarity))

def filter(query, index_list, tokenizer, model):
    _score = 0
    target_index = -1
    allintent = load_intent()
    intent_list = [allintent[i] for i in index_list]

    all_score = [score_query_and_intent_pretrained(query, intent, tokenizer, model) for intent in intent_list]
    all_score = torch.tensor(all_score)

    if len(intent_list) >= 2:
        top2_list, top2_indices = torch.topk(all_score, k=2)
    else:
        top2_list, top2_indices = torch.topk(all_score, k=1)
        top2_list = torch.tensor([top2_list.item(), top2_list.item()])
        top2_indices = torch.tensor([top2_indices.item(), top2_indices.item()])

    top1_score = top2_list[0]
    top2_score = top2_list[1]

    top1_index = index_list[top2_indices[0]]
    top2_index  = index_list[top2_indices[1]]

    top1_intent = intent_list[top2_indices[0]]
    top2_intent = intent_list[top2_indices[1]]

    print("Top1 score = {}, Top1 index = {}, Top1 intent = {}".format(top1_score, top1_index, top1_intent))
    print("Top2 score = {}, Top2 index = {}, Top2 intent = {}".format(top2_score, top2_index, top2_intent))
    return (top1_score, top1_index), (top2_score, top2_index)
    # for index, intent in enumerate(intent_list):
    #     s = score_query_and_intent_pretrained(query, intent, tokenizer, model)
    #     if s >= _score:
    #         _score = s
    #         target_index = index_list[index]
    #
    # return _score, target_index

def testing():
    my_index = "word2vec_index"
    my_doc = "my_doc"
    device = torch.device(0)
    tokenizer, model = load_finetune_model()

    testdata = load_test()
    print('数据集数量为{}'.format(len(testdata)))

    top1_right = 0
    top2_right = 0
    top1_wrong_list = []
    top2_wrong_list = []
    # matrix = np.zeros((24,24))
    top1_hit = np.zeros(63)
    top2_hit = np.zeros(63)
    label_num = np.zeros(63)

    for i, data in enumerate(testdata):
        sample = data[1]
        id = int(data[0])
        label_num[id] += 1

        result, t = keywordSearch(keywords1=sample, my_index=my_index, my_doc=my_doc)

        if len(result) == 0:
            continue

        re_label = [item[0] for item in result]

        # _score, target_index = filter(sample, re_label, tokenizer, model)
        print("Begin filtering ...")
        start_time = time.time()
        top1_info, top2_info = filter(sample, re_label, tokenizer, model)

        print("End filtering ...")
        end_time = time.time()
        print("Filter time = {}".format(end_time-start_time))

        top1_score, top1_index = top1_info[0], top1_info[1]
        top2_score, top2_index = top2_info[0], top2_info[1]
        # print(_score, " ", target_index, "\n")

        # matrix[id, top1_index] += 1

        if id in [top1_index]:
            top1_right += 1
            top1_hit[id] += 1
        else:
            top1_wrong_list.append({'data':i, 'text':sample, 'ans':id, 'target':[top1_index]})

        if id in [top1_index, top2_index]:
            top2_right += 1
            top2_hit[id] += 1
        else:
            top2_wrong_list.append({"data":i, "text":sample, "ans":id, "target":[top1_index, top2_index]})

        print("Top1 ---- {}".format(id in [top1_index]))
        print("Top2 ---- {}\n".format(id in [top1_index, top2_index]))


    # tr_matrix = [matrix[i,i] for i in range(24)]
    # assert sum(tr_matrix) == top1_right
    # print('混淆矩阵为 {}'.format(matrix))

    print(label_num)
    print(top1_hit)
    print(top2_hit)
    top1_hit_rate = top1_hit / label_num
    top2_hit_rate = top2_hit / label_num
    assert np.sum(label_num) == len(testdata)

    # r = [matrix[i,i] / np.sum(matrix[i,:]) for i in range(24)]
    print("Top1 recall = {}".format(top1_hit_rate))
    print('Top1 mean recall = {}'.format(np.sum(top1_hit_rate) / 63))
    print('Top1 acc = {}'.format(top1_right / len(testdata)))

    print("Top2 recall = {}".format(top2_hit_rate))
    print('Top2 mean recall = {}'.format(np.sum(top2_hit_rate) / 63))
    print('Top2 acc = {}'.format(top2_right / len(testdata)))

    return top1_right, top2_right, top1_wrong_list, top2_wrong_list

if __name__ == "__main__":
    testing()
# intent_list = load_intent()
# intent_list[-1] = "还款 还钱 还款期 还款额 还款率 还款日"
# for intent in intent_list:
#     demo("还款流程是怎样的", intent)

# demo("还款流程是怎样的", "还款 还钱 还款期 还款额 还款率 还款日")
# demo("还款流程是怎样的", "怎么办理")
# demo("还款流程是怎样的", "有信用卡了")
# demo("还款流程是怎样的", "此信用卡消费多久还款")
