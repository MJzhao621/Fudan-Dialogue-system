import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import copy
import random
import numpy as np
from apex import amp
from transformers import AdamW, BertTokenizer, BertModel, get_linear_schedule_with_warmup
import json

def load_intent(intentpath=r"/remote-home/gzhao/retrival-engine/sales/data/corpus1.0/intent_list_corpus1.0.json"):
    fjson = open(intentpath, "r", encoding="utf-8")
    intent = json.load(fjson)
    return intent

def load_data(datapath=r"/remote-home/gzhao/retrival-engine/sales/data/corpus1.0/data.txt"):
    data = []
    f = open(datapath, encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        line = line[:-1]
        data.append((line.split('\t')[0], line.split('\t')[1]))
    return data

def create_query_and_intent(data, intent):
    output = []
    for d in data:
        output.append((d[1], intent[int(d[0])]))
    return output

def load_chinese_bert_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained(r'/remote-home/gzhao/retrival-engine/sales/chinese-bert')
    model = BertModel.from_pretrained(r'/remote-home/gzhao/retrival-engine/sales/chinese-bert')
    return tokenizer, model

def create_dataset(data, intent, tokenizer):
    def pad_tokens(ids, max_length):
        pad_ids = ids + [0]*(max_length-len(ids))
        attention_mask = [1]*len(ids)+[0]*(max_length-len(ids))
        assert len(pad_ids) == len(attention_mask) == max_length
        return pad_ids, attention_mask

    def create_tensordataset(output):
        query = [tokenizer.encode(o[0], add_special_tokens=True) for o in output]
        query_max_length = max([len(o) for o in query])
        query = [pad_tokens(o, query_max_length) for o in query]
        query_tensor = torch.tensor([o[0] for o in query])
        query_attention_mask = torch.tensor([o[1] for o in query])

        intenting = [tokenizer.encode(o[1], add_special_tokens=True) for o in output]
        intent_max_length = max([len(o) for o in intenting])
        intenting = [pad_tokens(o, intent_max_length) for o in intenting]
        intent_tensor = torch.tensor([o[0] for o in intenting])
        intent_attention_tensor = torch.tensor([o[1] for o in intenting])

        return TensorDataset(query_tensor, query_attention_mask, intent_tensor, intent_attention_tensor)

    output = create_query_and_intent(data, intent)
    random.shuffle(output)
    dataset = create_tensordataset(output)

    train_num = int(0.8 * len(output))
    valid_num = int(0.1 * len(output))
    test_num = len(output) - train_num - valid_num
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_num, valid_num, test_num])

    return dataset, valid_dataset, test_dataset

class Cross_Encoder(torch.nn.Module):
    def __init__(self, model, device):
        super(Cross_Encoder, self).__init__()
        self.model_name = "cross-encoder"
        self.device = device
        self.bert_model = model
        self.linear = torch.nn.Linear(768, 1)

    def score(self, query, intent, tokenizer):
        query_ids = torch.tensor(tokenizer.encode(query)).unsqueeze(0).to(self.device)
        intent_ids = torch.tensor(tokenizer.encode(intent)).unsqueeze(0).to(self.device)
        input_ids = torch.cat([query_ids, intent_ids], dim=1)
        output = self.bert_model(input_ids)[0]
        output_hidden = torch.mean(output, axis=1)
        score_hidden = self.linear(output_hidden)
        return score_hidden.item()

    def forward(self, batch_query_ids, batch_query_attention, batch_intent_ids, batch_intent_attention):
        batch_size = batch_query_ids.size(0)
        output_score_list = []
        for i in range(batch_size):
            ids_repeat = batch_query_ids[i,:].repeat(batch_size, 1)
            attention_repeat = batch_query_attention[i,:].repeat(batch_size, 1)
            input_ids = torch.cat((ids_repeat, batch_intent_ids), dim=1)
            input_attention = torch.cat((attention_repeat, batch_intent_attention), dim=1)
            output = self.bert_model(input_ids, input_attention)[0]
            output_hidden = torch.mean(output, axis=1)
            assert output_hidden.size() == (batch_size, 768)
            score_hidden = self.linear(output_hidden)
            # del output, output_hidden, ids_repeat, attention_repeat, input_ids, input_attention
            output_score_list.append(score_hidden.T)

        similarity_matrix = torch.stack(output_score_list, dim=1).squeeze(0)
        similarity_matrix = F.log_softmax(similarity_matrix, dim=1)
        assert similarity_matrix.size() == (batch_size, batch_size)

        values, indices = torch.max(similarity_matrix, axis=1)
        ok = torch.sum(indices == torch.arange(batch_size).to(self.device))
        I = torch.eye(batch_size).to(self.device)
        loss = torch.sum(similarity_matrix * I)
        return -loss, ok

class Bi_Encoder(torch.nn.Module):
    def __init__(self, model, device):
        super(Bi_Encoder, self).__init__()
        self.model_name = "bi-encoder"
        self.bert_model = model
        self.device = device

    def score(self, query, intent, tokenizer):
        query_ids = torch.tensor(tokenizer.encode(query)).unsqueeze(0).to(self.device)
        intent_ids = torch.tensor(tokenizer.encode(query)).unsqueeze(0).to(self.device)
        query_hidden = self.bert_model(query_ids)[0]
        intent_hidden = self.bert_model(intent_ids)[0]
        query_hidden = torch.mean(query_hidden, axis=1)
        intent_hidden = torch.mean(intent_hidden, axis=1)
        score_hidden = torch.matmul(query_hidden, intent_hidden.T)
        return score_hidden.item()

    def forward(self, batch_query_ids, batch_query_attention, batch_intent_ids, batch_intent_attenton):
        query_output = self.bert_model(batch_query_ids, attention_mask=batch_query_attention)[0]
        intent_output = self.bert_model(batch_intent_ids, attention_mask=batch_intent_attenton)[0]
        # print(query_output.size())
        # print(intent_output.size())

        query_hidden = torch.mean(query_output, axis=1)
        intent_hidden = torch.mean(intent_output, axis=1)
        # print(query_hidden.size())
        # print(intent_hidden.size())

        similarity_matrix = torch.matmul(query_hidden, intent_hidden.T)
        similarity_matrix = F.log_softmax(similarity_matrix, dim=1)
        batch_size = similarity_matrix.size(0)

        values, indices = torch.max(similarity_matrix, axis=1)
        ok = torch.sum(indices == torch.arange(batch_size).to(self.device))
        I = torch.eye(batch_size).to(self.device)
        loss = torch.sum(similarity_matrix * I)
        return -loss, ok


def main(model_type = "bi-encoder", save_index = "01"):
    no_decay = ["bias", "LayerNorm.weight"]
    weight_decay = 0
    fp16 = 1
    fp16_opt_level = "O1"
    test_mode = 0
    warmup_steps = 0
    epoches = 100
    use_gpu = 1
    save_model_path = r"/remote-home/gzhao/retrival-engine/sales/save/bert-finetune-crossencoder-" + save_index + ".pt"
    # local_save_model_path = r"D:\PycharmProject\Project1\retrival-engine\sales\bert-finetune-bi-encoder.pt"

    if use_gpu:
        device = torch.device(0)
    else:
        device = torch.device("cpu")

    print("Loading dataset...")
    intent = load_intent()
    data = load_data()
    tokenizer, model = load_chinese_bert_model_and_tokenizer()
    train_dataset, valid_dataset, test_dataset = create_dataset(data, intent, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=10, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    print("Length of train dataloader is {}".format(len(train_dataloader)))
    print("Length of valid dataloader is {}".format(len(valid_dataloader)))
    print("Length of test dataloader is {}".format(len(test_dataloader)))

    valid_iter = tqdm(valid_dataloader, desc="Iteration")

    print("Loading similarity model...")
    if model_type == "bi-encoder":
        similarity = Bi_Encoder(model=model, device=device).to(device)
    elif model_type == "cross-encoder":
        similarity = Cross_Encoder(model=model, device=device).to(device)
    else:
        raise ValueError("No such model type")
    print("Similarity Model is {}".format(similarity.model_name))

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in similarity.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in similarity.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1.5e-5, eps=1e-8)

    if fp16:
        similarity, optimizer = amp.initialize(similarity, optimizer, opt_level=fp16_opt_level)

    t_total = len(train_dataloader) * epoches
    if not test_mode:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

    last_valid_acc = 0.0


    for epoch in range(epoches):
        print("Epoch {}".format(epoch))
        similarity.train()
        optimizer.zero_grad()
        ok_num = 0
        total_num = 0
        total_loss = 0.0
        train_iter = tqdm(train_dataloader, desc="Iteration")
        for i, batch in enumerate(train_iter):
            batch = tuple(t.to(device) for t in batch)
            batch_query_emb, batch_query_attention = batch[0], batch[1]
            batch_intent_emb, batch_intent_attention = batch[2], batch[3]

            loss, ok = similarity(batch_query_emb, batch_query_attention, batch_intent_emb, batch_intent_attention)

            ok_num += ok.item()
            total_num += batch_query_emb.size(0)

            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()
            total_loss += loss.item()

        acc = ok_num / total_num
        ave_loss = total_loss / len(train_dataloader)
        print("Epoch {}, train loss = {}, train accuracy = {}".format(epoch, ave_loss, acc))

        # eval the similarity model
        valid_ok_num = 0
        valid_total_num = 0
        valid_total_loss = 0.0
        similarity.eval()
        for i, batch in enumerate(valid_iter):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                valid_query_emb, valid_query_attention = batch[0], batch[1]
                valid_intent_emb, valid_intent_attention = batch[2], batch[3]
                valid_loss, valid_ok = similarity(valid_query_emb, valid_query_attention,
                                                   valid_intent_emb, valid_intent_attention)
                batch_size = valid_intent_emb.size(0)
                valid_ok_num += valid_ok
                valid_total_num += batch_size
                valid_total_loss += valid_loss

        valid_acc = valid_ok_num.item() / valid_total_num
        valid_mean_loss = valid_total_loss / len(valid_dataloader)
        print("Epoch {}, valid loss = {}, valid accuracy = {}".format(epoch, valid_mean_loss, valid_acc))

        if epoch > 30 and valid_acc < last_valid_acc:
            break
        else:
            print("Saving model to ", save_model_path)

            saved_model = {}
            for k, v in similarity.state_dict().items():
                saved_model[k] = v.cpu()

            torch.save(saved_model, save_model_path)

        last_valid_acc = valid_acc

    # query = "请问信用卡是怎么收费的"
    # intent = "收费"
    # print("Score of {} is {}".format(similarity.model_name, similarity.score(query, intent, tokenizer)))




if __name__ == "__main__":
    main(model_type="cross-encoder", save_index="02")
