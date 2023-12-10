import re
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModel

def inverse_tokenize(tokens):
    r"""
    Convert tokens to sentence.
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    Watch out!
    Default punctuation add to the word before its index,
    it may raise inconsistency bug.
    :param list[str]r tokens: target token list
    :return: str
    """
    assert isinstance(tokens, list)
    text = ' '.join(tokens)
    step1 = text.replace("`` ", '"') \
        .replace(" ''", '"') \
        .replace('. . .', '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
        "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    step7 = step6.replace('do nt', 'dont').replace('Do nt', 'Dont')
    step8 = step7.replace(' - ', '-')
    return step8.strip()

def mark_fewrel_entity(new_pos, new_entity_h, new_entity_t, sent_len):
    mark_head = np.array([0] * sent_len) 
    mark_tail = np.array([0] * sent_len)
    mark_head[new_entity_h[0]:new_entity_h[1]] = 1 # mark head entity, which is between [E1] and [E1/]
    mark_tail[new_entity_t[0]:new_entity_t[1]] = 1 # mark head entity, which is between [E2] and [E2/]
    marked_e1 = np.array([0] * sent_len)
    marked_e2 = np.array([0] * sent_len)
    marked_e1[new_pos[0]] = 1 # mark [E1]
    marked_e2[new_pos[1]] = 1 # mark [E2]
    return torch.tensor(marked_e1), torch.tensor(marked_e2), \
             torch.tensor(mark_head), torch.tensor(mark_tail)

def pad_or_truncate(tensor, target_width):
    current_width = tensor.size(0)
    if current_width < target_width:
        pad_size = target_width - current_width
        padding = torch.zeros(pad_size, dtype=tensor.dtype, device=tensor.device)
        padded_tensor = torch.cat((tensor, padding), dim=0)
        return padded_tensor
    elif current_width > target_width:
        truncated_tensor = tensor[:target_width]
        return truncated_tensor
    else:
        return tensor

class Dataset(Dataset):

    def __init__(self, mode, data_file, description_file, description_file_processed, m, pretrained_model_name_or_path, max_len, model, args, use_mlm = False, expand_or_not = True):
        '''
        data_file: dataset path
        description_file: relation description file path
        description_file_processed: RE-matching description file path
        m: the number of unseen relations
        '''
        super(Dataset, self).__init__()
        self.data_types = ["train", "dev", "test"]
        assert mode in self.data_types
        self.mode = mode # train, dev, test
        self.data_file = data_file # path
        self.description_file = description_file # path
        self.description_file_processed = description_file_processed # path
        self.m = m # 5/10/15/...
        self.pretrained_model_name_or_path = pretrained_model_name_or_path # bert-base-uncased or others
        self.max_len = max_len # seq max length
        self.use_mlm = use_mlm # use mlm expend entitys or not
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path) # "../bert-base-uncased"
        # self.tokenizer = AutoTokenizer.from_pretrained('deberta-v3-large')
        self.label_ids = {} # {"train":[str,...],"dev":[str,...],"test":[str,...],}
        self.descriptions = {} # {"train": [sample],...}
        self.data = {} # {"train": [sample],...}
        self.data_features = {} # {"train": [{"input_ids": tensor, "att_mask":tensor,"entity_idx":tensor, "label":},...], ...}
        self.des_features = {} # {"train":{"rid":[contex_emb, h_entity_emb, t_entity_emb],...},...}

        # if self.pretrained_model in ['bert-base-uncased', 'distilbert-base-uncased']:
        #     self.head_mark_ids = 1001
        #     self.tail_mark_ids = 1030
        # elif self.pretrained_model in ['roberta-base', 'distilroberta-base']:
        #     self.head_mark_ids = 849
        #     self.tail_mark_ids = 787
        # 先默认使用BERT
        self.head_mark_ids = 1001 # #对应的token id，在prlm的vocab.txt里面定义
        self.tail_mark_ids = 1030 # @同上
        self.model = model
        self.args = args

        self.read_data()
        if(expand_or_not):
            self.expand_data()
        # self.convert_data_and_des_to_features()
        self.convert_data_and_des_to_features_DSSM()

    def split_labels(self):
        # 'random split the labels: train dev test'
        # all_ids = []
        # with open(self.description_file, 'r', encoding='utf-8') as d:
        #     id2description = json.load(d)
        #     all_ids = list(id2description.keys())
        # # print(type(all_ids))
        # random.shuffle(all_ids)
        
        # self.label_ids["train"] = all_ids[:len(all_ids)-2*self.m] # [:70]
        # self.label_ids["dev"] = all_ids[len(all_ids)-2*self.m:len(all_ids)-self.m] # [70:75]
        # self.label_ids["test"] = all_ids[len(all_ids)-self.m:] # [75:]
        # # print("unseen: " + str(self.m))
        # print(self.label_ids["train"])
        # print(self.label_ids["dev"])
        # print(self.label_ids["test"])

        with open('/home/lsl/projects/ZeroRE/RE-Matching/data/rel2id/fewrel_rel2id/fewrel_rel2id_10_42.json', 'r', encoding='utf-8') as r2id:
            relation2idx = json.load(r2id)
            train_relation2idx,dev_relation2idx, test_relation2idx = relation2idx['train'], relation2idx['eval'], relation2idx['test']
        #     train_idx2relation, dev_idx2relation, test_relation2idx = dict((v, k) for k, v in train_relation2idx.items()), \
        #                                             dict((v, k) for k, v in dev_relation2idx.items()), \
        #                                             dict((v, k) for k, v in test_relation2idx.items())
        # print(dev_idx2relation.keys(),test_relation2idx.keys())

        self.label_ids["train"], self.label_ids["dev"], self.label_ids["test"] = list(k for k, v in train_relation2idx.items()), \
                                                    list(k for k, v in dev_relation2idx.items()), \
                                                    list(k for k, v in test_relation2idx.items())

        print(self.label_ids["train"])
        print(self.label_ids["dev"])
        print(self.label_ids["test"])
    
    def read_data(self):
        self.split_labels()

        # load data
        with open(self.data_file, 'r', encoding='utf-8') as d:
            raw_data = json.load(d)
            for t in self.data_types:
                self.data[t] = [i for i in raw_data if i['relation'] in self.label_ids[t]]

        # load relation_description
        with open(self.description_file_processed, 'r', encoding='utf-8') as rd:
            relation_desc = json.load(rd)
            for t in self.data_types:
                self.descriptions[t] = [i for i in relation_desc if i['relation'] in self.label_ids[t]]
            
        # print('there are {} kinds of relation in test.'.format(len(set(self.label_ids["test"]))))
        # print('the lengths of test data is {} '.format(len(self.data["test"])))

        print(f'train data numbers: {len(self.data["train"])}')
        print(f'dev data numbers: {len(self.data["dev"])}')
        print(f'test data numbers: {len(self.data["test"])}')

    def convert_data_and_des_to_features_DSSM(self):
        # convert data to features
        for t in self.data_types:
            data = self.data[t]
            self.data_features[t] = []
            for sample in tqdm(data, "convert data to features: "): 
                pos1 = sample['h']['pos'][0]
                pos1_end = sample['h']['pos'][1]
                pos2 = sample['t']['pos'][0]
                pos2_end = sample['t']['pos'][1]
                words = sample['token']
                # sentence:  xxx xxx xxx # head_entity # xxx xxx xxx @ tail_entity @ xxx xxx xxx
                if pos1 < pos2:
                    new_words = words[:pos1] + ['#'] + words[pos1:pos1_end] + ['#'] + words[pos1_end:pos2] \
                                + ['@'] + words[pos2:pos2_end] + ['@'] + words[pos2_end:]
                else:
                    new_words = words[:pos2] + ['@'] + words[pos2:pos2_end] + ['@'] + words[pos2_end:pos1] \
                                + ['#'] + words[pos1:pos1_end] + ['#'] + words[pos1_end:]
                
                sentence = " ".join(new_words)
                tokens_info = self.tokenizer(sentence)
                input_ids = tokens_info['input_ids']
                attention_mask = torch.tensor(tokens_info['attention_mask'])

                new_head_pos = input_ids.index(self.head_mark_ids)
                new_tail_pos = input_ids.index(self.tail_mark_ids)
                new_head_end_pos = input_ids.index(self.head_mark_ids, new_head_pos + 1)
                new_tail_end_pos = input_ids.index(self.tail_mark_ids, new_tail_pos + 1)
                new_pos = (new_head_pos, new_tail_pos)
                new_entity_h = (new_head_pos + 1, new_head_end_pos)
                new_entity_t = (new_tail_pos + 1, new_tail_end_pos)

                marked_e1, marked_e2, mark_head, mark_tail = mark_fewrel_entity(new_pos, new_entity_h, new_entity_t,
                                                                                len(input_ids))
                
                head_range = torch.tensor(new_entity_h)
                tail_range = torch.tensor(new_entity_t)
                input_ids = torch.tensor(input_ids)

                rid_num = int(sample['relation'][1:]) # i.e. P191 ...
                rid_tensor = torch.tensor([rid_num])

                for des in self.descriptions[t]:
                    if des['relation'] == sample['relation']:
                        des_sentence = inverse_tokenize(des['description'])
                        des_input_ids = torch.tensor(self.tokenizer(des_sentence)['input_ids'])
                        des_attention_mask = torch.tensor(self.tokenizer(des_sentence)['attention_mask'])

                sample_features = {
                    "input_ids": pad_or_truncate(input_ids, self.max_len),
                    "attention_mask": pad_or_truncate(attention_mask, self.max_len),
                    "des_input_ids": pad_or_truncate(des_input_ids, self.max_len),
                    "des_attention_mask": pad_or_truncate(des_attention_mask, self.max_len),
                    "marked_e1": pad_or_truncate(marked_e1, self.max_len),
                    "marked_e2": pad_or_truncate(marked_e2, self.max_len),
                    "mark_head": pad_or_truncate(mark_head, self.max_len),
                    "mark_tail": pad_or_truncate(mark_tail, self.max_len),
                    "head_range": pad_or_truncate(head_range, self.max_len),
                    "tail_range": pad_or_truncate(tail_range, self.max_len),
                    "rid": pad_or_truncate(rid_tensor, self.max_len),
                }
                self.data_features[t].append(sample_features)
                
        # convert descriptions to features
        for t in self.data_types:
            self.des_features[t] = {}
            des = self.descriptions[t]
            des_sentences = [inverse_tokenize(sample['description']) for sample in des]
           
            # get description's context embeddings
            des_sentence_input_ids = [self.tokenizer(sent)['input_ids'] for sent in des_sentences] # sentence embedding 768维  ndarray 70x768
            des_sentence_attention_masks = [self.tokenizer(sent)['attention_mask'] for sent in des_sentences]

            # get relation id
            des_relation_ids = [sample['relation'] for sample in des]
            for rid, input_ids, attention_mask in zip(des_relation_ids, des_sentence_input_ids, des_sentence_attention_masks):        
                input_ids = pad_or_truncate(torch.tensor(input_ids), self.max_len)
                attention_mask = pad_or_truncate(torch.tensor(attention_mask), self.max_len)
                self.des_features[t][rid] = torch.cat((input_ids, attention_mask), dim=0)

    def __getitem__(self, index):
            return self.data_features[self.mode][index]
    
    # def get_des_features_by_rid(self, rid_num:int):
    #     rid = 'P' + str(int(rid_num))
    #     return self.des_features[self.mode][rid]
    
    def get_evaluate_des_features(self):
        rids = self.label_ids[self.mode]
        return torch.stack([self.des_features[self.mode][rid] for rid in rids])
    
    def convert_rid_to_label(self, rid):
        return self.label_ids[self.mode].index(rid)
    
    def __len__(self):
        return len(self.data[self.mode])