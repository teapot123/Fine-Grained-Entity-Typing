import torch
from transformers import RobertaTokenizer
from hierarchy import *
from transformers import pipeline
unmasker = pipeline('fill-mask', model='/shared/data2/jiaxinh3/Typing/pre-trained', device=0)
tokenizer = RobertaTokenizer.from_pretrained('/shared/data2/jiaxinh3/Typing/pre-trained')

def save_to_file(sents):
    with open('tmp.txt', 'a') as fout:
        for line in sents:
            fout.write(line+'\n')


def tokenize_line(pos_masks, pos_labels, good_sent, bad_sent, entity_list, node_list):
    max_seq_len=128
    pos_masks = tokenizer(pos_masks, return_tensors='pt', max_length=max_seq_len, padding='max_length', truncation=True)
    pos_labels = tokenizer(pos_labels, return_tensors='pt', max_length=max_seq_len, padding='max_length', truncation=True)
    masked_id = torch.nonzero(pos_masks["input_ids"] == tokenizer.mask_token_id, as_tuple=False)[:, 1].unsqueeze(1)
    if len(masked_id) != len(pos_masks["input_ids"]):
        bad_sent += len(masked_id)
        return good_sent, bad_sent, []
    else:
        good_sent += 1 
        train_sent = pos_masks["input_ids"]
        sent_att = pos_masks["attention_mask"]
        sent_mask = masked_id
        sent_label = pos_labels["input_ids"]
        label_ids = []
        for _, entity_type in entity_list:
            if node_list == None:
                label_id = tokenizer.encode(' '+entity_type)[1]
            else:
                label_id = node_list.index(entity_type)
            label_ids.append(label_id)
        label_ids = torch.tensor(label_ids, dtype=torch.int64)
        return good_sent, bad_sent, [train_sent, sent_att, sent_mask, sent_label, label_ids]


def process_data(ner_tags, sents, tokenizer, filename='', max_seq_len=128):
    print("Start processing raw data!")
    good_sent = 0
    bad_sent = 0
    train_sent, sent_mask, sent_att, sent_label, label_ids = [], [], [], [], []
    neg_train_sent, neg_sent_mask, neg_sent_att, neg_sent_label = [], [], [], []
    parent_dict = {}

    if filename != '':
        child_dict, parent_dict, level_dict = extract_hierarchy(filename)
        node_list = [node for node in parent_dict if len(parent_dict[node][node]) == 1]
        node_id_list = [tokenizer.encode(" "+node)[1] for node in node_list]
        # node_id_list = [x for x in range(len(tokenizer.get_vocab()))]


    for k,line in enumerate(sents):
        sent = line.strip()
        words = sent.split(' ')
        tags = ner_tags[k].strip().split()
        assert len(tags) == len(words)
        entity_list, pos_masks, pos_labels = extract_entity(words, tags, parent_dict)

        if len(pos_masks) == 0:
            continue
        # save_to_file(pos_labels)
        
        good_sent, bad_sent, positive_data = tokenize_line(pos_masks, pos_labels,\
            good_sent, bad_sent, entity_list, node_list)
        # good_sent, bad_sent, negative_data = tokenize_line(neg_masks, neg_labels, good_sent, bad_sent)
        train_sent.append(positive_data[0])
        sent_att.append(positive_data[1])
        sent_mask.append(positive_data[2])
        sent_label.append(positive_data[3])
        label_ids.append(positive_data[4])
        # neg_train_sent.append(negative_data[0])
        # neg_sent_mask.append(negative_data[1])
        # neg_sent_att.append(negative_data[2])
        # neg_sent_label.append(negative_data[3])
    
    train_sent = torch.cat(train_sent, dim=0)
    sent_att = torch.cat(sent_att, dim=0)
    sent_mask = torch.cat(sent_mask, dim=0)
    sent_label = torch.cat(sent_label, dim=0)
    label_ids = torch.cat(label_ids, dim=0)
    node_id_list = torch.tensor(node_id_list, dtype=torch.int64)
    # neg_train_sent = torch.cat(neg_train_sent, dim=0)
    # neg_sent_att = torch.cat(neg_sent_att, dim=0)
    # neg_sent_mask = torch.cat(neg_sent_mask, dim=0)
    # neg_sent_label = torch.cat(neg_sent_label, dim=0)
    print(f"train_sent: {train_sent.shape} label_ids: {label_ids.shape} node_id_list: {node_id_list}")
    print(f"truncate ratio: {bad_sent/(bad_sent+good_sent)}")
    print(f"good sent: {good_sent} bad sent: {bad_sent} total: {bad_sent+good_sent}")
    return [train_sent, sent_att, sent_mask, sent_label, label_ids, node_id_list], \
        [neg_train_sent, neg_sent_att, neg_sent_mask, neg_sent_label]


def tokenize_text(pos_masks, pos_labels):
    max_seq_len=128
    pos_masks = tokenizer(pos_masks, return_tensors='pt', max_length=max_seq_len, padding='max_length', truncation=True)
    pos_labels = tokenizer(pos_labels, return_tensors='pt', max_length=max_seq_len, padding='max_length', truncation=True)
    masked_id = torch.nonzero(pos_masks["input_ids"] == tokenizer.mask_token_id, as_tuple=False)[:, 1].unsqueeze(1)
    train_sent = pos_masks["input_ids"]
    sent_att = pos_masks["attention_mask"]
    sent_mask = masked_id
    sent_label = pos_labels["input_ids"]
    return [train_sent, sent_att, sent_mask, sent_label]


def process_hierarchy(filename):
    pos_masks, pos_labels, neg_masks, neg_labels = extract_relations_from_hierarchy(filename)
    pos_data = tokenize_text(pos_masks, pos_labels)
    neg_data = tokenize_text(neg_masks, neg_labels)
    print(f"number of pos samples: {len(pos_data[0])} neg: {len(neg_data[0])}")
    return pos_data, neg_data


def read_files(train_text, train_ner):
    with open(train_ner) as fner, open(train_text) as f:
        train_ner_tags, train_words = fner.readlines(), f.readlines() 
    return train_words, train_ner_tags
    
def generate_batch(batch):
    train_sent, sent_att, sent_mask, sent_label = batch
    return train_sent, sent_att, sent_mask, sent_label

import json
from collections import defaultdict
def read_types(filename):
    train_type = defaultdict(int)
    
    single_label = 0
    multi_label = 0
    with open(filename) as f:
        for line in f:
            d = json.loads(line.strip())
            cats = d['y_category']
            finest_cat = ''
            for cat in cats:
                if len(finest_cat) == 0:
                    finest_cat = cat
                elif finest_cat in cat:
                    finest_cat = cat
                elif (finest_cat not in cat) and (cat not in finest_cat):
                    finest_cat = ''
                    break
            if finest_cat == '':
                multi_label += 1
                continue
            else:
                single_label += 1
                train_type[finest_cat] += 1
                
    print(filename)
    print(f"multi-label: {multi_label} single-label: {single_label} train_type: {len(train_type)}")
    # print(train_type)

    # with open(hier_name) as f:
    #     for line in f:
    #         type_name = line.strip()
    #         if type_name not in train_type:
    #             print(f"{type_name}:0")
    return [k for k,v in train_type.items()]


def read_file_and_hier(filename, new_hier, old_hier, train_type_count, use_node_list = True, \
    extract_new_instance=False):
    train_types = {}
    reverse_train_types = {}
    child_dict, parent_dict, level_dict, relation_list = extract_hierarchy(new_hier)
    new_instances = defaultdict(dict)
    
    with open(old_hier) as f:
        for line in f:
            if (len(line.strip().split(':')) < 2):
                continue
            tmp = line.strip().split(':')
            if tmp[0] not in train_type_count:
                continue
            train_types[tmp[0]] = tmp[1]
            reverse_train_types[tokenizer.encode(" "+tmp[1])[1]] = tmp[0].split('/')[1:]
    
    if use_node_list:
        node_list = [node for node in train_types.values()]
        node_id_list = [tokenizer.encode(" "+node)[1] for node in train_types.values()]
    else:
        node_list = None
        node_id_list = [x for x in range(len(tokenizer.get_vocab()))]

    r_list = []
    for k in relation_list:
        child = tokenizer.encode(' '+k)[1]
        parent = tokenizer.encode(' '+relation_list[k])[1]
        if child in node_id_list and parent in node_id_list:
            r_list.append([node_id_list.index(child), node_id_list.index(parent)])
    print(f"r list: {r_list}")
    
    good_sent = 0
    bad_sent = 0
    replace_sent = 0
    train_sent, sent_mask, sent_att, sent_label, label_ids = [], [], [], [], []
    total_words, total_entity_list = [], []

    with open(filename) as f:
        for line in f:
            d = json.loads(line.strip())
            cats = d['y_category']
            finest_cat = ''
            for cat in cats:
                if len(finest_cat) == 0:
                    finest_cat = cat
                elif finest_cat in cat:
                    finest_cat = cat
                elif (finest_cat not in cat) and (cat not in finest_cat):
                    finest_cat = ''
                    break
            if finest_cat == '' or (finest_cat not in train_type_count):
                continue
            if finest_cat not in train_types:
                continue
            category = train_types[finest_cat]

            words = d['left_context'] + d['mention_as_list'] + d['right_context']

            tags = ['O'] * len(d['left_context']) + ['B-'+category] \
                + ['I-'+category] * (len(d['mention_as_list']) - 1) \
                + ['O'] * len(d['right_context'])
            if len(words) != len(tags):
                continue

            entity_list, pos_masks, pos_labels = extract_entity(words, tags, parent_dict)
            if extract_new_instance:
                new_instances = extract_instance(words, entity_list, unmasker, new_instances)

            if len(pos_masks) == 0:
                continue
            
            good_sent, bad_sent, positive_data = tokenize_line(pos_masks, pos_labels,\
                good_sent, bad_sent, entity_list, node_list)
            if len(positive_data) == 0:
                continue
            train_sent.append(positive_data[0])
            sent_att.append(positive_data[1])
            sent_mask.append(positive_data[2])
            sent_label.append(positive_data[3])
            label_ids.append(positive_data[4])
            total_words.append(words)
            total_entity_list.append(entity_list)
    
    train_sent = torch.cat(train_sent, dim=0)
    sent_att = torch.cat(sent_att, dim=0)
    sent_mask = torch.cat(sent_mask, dim=0)
    sent_label = torch.cat(sent_label, dim=0)
    label_ids = torch.cat(label_ids, dim=0)
    node_id_list = torch.tensor(node_id_list, dtype=torch.int64)

    print(f"train_sent: {train_sent.shape} label_ids: {label_ids.shape} node_id_list: {node_id_list}")
    print(f"truncate ratio: {bad_sent/(bad_sent+good_sent)}")
    print(f"good sent: {good_sent} bad sent: {bad_sent} total: {bad_sent+good_sent}")
    print(f"replace sent: {replace_sent}")
    

    new_instances_id = {}
    for entity_type in new_instances:
        new_instances_id[tokenizer.encode(' '+entity_type)[1]] = \
            {tokenizer.encode(k)[1]:v for k,v in new_instances[entity_type].items()}
    return [train_sent, sent_att, sent_mask, sent_label, label_ids, node_id_list, reverse_train_types,\
        new_instances_id, r_list, [total_words, total_entity_list]]
