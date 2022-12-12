import numpy as np
from nltk.corpus import stopwords
import string

stops = set(stopwords.words('english'))

def extract_pos(words, entity_list):
    pos_masks = []
    pos_labels = []
    for entity_name, entity_type in entity_list:
        
        pos_mask = words  + entity_name + 'is a <mask> .'.split(' ')
        pos_label = words + entity_name + 'is a'.split(' ') + [entity_type] + ['.']

        pos_masks.append(pos_mask)
        pos_labels.append(pos_label)
    return pos_masks, pos_labels


def extract_instance(words, entity_list, unmasker, new_instances):
    for entity_name, entity_type in entity_list:
        pos_mask = words + entity_name + ', as well as <mask> , is a'.split(' ') + [entity_type] + ['.']
        pos_mask = ' '.join(pos_mask)
        word_types = {x['token_str']:x['score'] for x in unmasker(pos_mask, top_k=5)}
        for k, v in word_types.items():
            if k.lower()[1:] in stops or all(j in string.punctuation for j in k[1:]):
                continue
            if k not in new_instances[entity_type] or new_instances[entity_type][k] < v:
                new_instances[entity_type][k] = v
    return new_instances


def count_level(line):
    i = 0
    while(i < len(line) and line[i] == '\t'):
        i += 1
    return i

def extract_hierarchy(filename):
    child_dict = {}
    parent_dict = {}
    level_dict = {}
    current_parent = {}
    relation_list = {}

    child_dict['name'] = 'root'
    with open(filename) as f:
        for line in f:
            level = count_level(line)
            level_dict[line.strip()] = level
            if level == 0:
                child_dict[line.strip()] = {}
                child_dict[line.strip()]['name'] = line.strip()
                parent_dict[line.strip()] = child_dict
                current_parent[level] = child_dict[line.strip()]
            else:
                parent_dict[line.strip()] = current_parent[level-1]
                parent_dict[line.strip()][line.strip()] = {}
                current_dict = parent_dict[line.strip()][line.strip()]
                current_dict['name'] = line.strip()
                current_parent[level] = current_dict
                relation_list[line.strip()] = current_parent[level-1]['name']

    return child_dict, parent_dict, level_dict, relation_list


def extract_neg_from_hierarchy(words, entity_list, parent_dict):
    neg_masks = []
    neg_labels = []
    for entity_name, entity_type in entity_list:
        parent_node = parent_dict[entity_type]
        siblings = [parent_node[s]['name'] for s in parent_node if \
            (s != 'name' and parent_node[s]['name'] != entity_type)]

        for s in siblings:
            neg_mask = words + 'The term'.split(' ') + entity_name + 'is a type of <mask> .'.split(' ')
            neg_label = words + 'The term'.split(' ') + entity_name + 'is a type of'.split(' ') + [s] + ['.']

            neg_masks.append(neg_mask)
            neg_labels.append(neg_label)
    return neg_masks, neg_labels


def extract_siblings(parent_dict):
    sibling_dict = {}
    for node in parent_dict:
        parent = parent_dict[node]
        sibling_dict[node] = [parent_dict[c]['name'] for c in parent_dict[node] if (c!='name' and parent_dict[c]['name'] != node)]
    return sibling_dict

def extract_neg(words, span_list):
    non_entity = '\\(\\'
    neg_masks = []
    neg_labels = []
    total_length = len(words)  # '.' is not needed.
    total_span = [i for i in range(total_length)]
    neg_span = []
    for start_index, end_index in span_list:
        span = end_index - start_index
        new_start = np.random.choice(total_span, size = min(5, total_length), replace=False).tolist()
        for s in new_start:
            if [s, s+span] not in span_list and (s+span) <= total_length and [s, s+span] not in neg_span :
                entity_name = words[s:s+span]
                
                neg_mask = words + 'The term'.split(' ') + entity_name + 'is a type of <mask> .'.split(' ')
                neg_label = words + 'The term'.split(' ') + entity_name + 'is a type of'.split(' ') + [non_entity] + ['.']     
               
                neg_masks.append(neg_mask)
                neg_labels.append(neg_label)
                neg_span.append([s, s+span])
    if len(span_list) == 0:
        for i in range(min(5, total_length)):
            s = np.random.choice(total_length, size = 1, replace=False).tolist()[0]
            span = max(np.random.choice(total_length - s, size = 1, replace=False).tolist()[0], 1)
            if [s, s+span] not in neg_span :
                entity_name = words[s:s+span]
                
                neg_mask = words + 'The term'.split(' ') + entity_name + 'is a type of <mask> .'.split(' ')
                neg_label = words + 'The term'.split(' ') + entity_name + 'is a type of'.split(' ') + [non_entity] + ['.']     
               
                neg_masks.append(neg_mask)
                neg_labels.append(neg_label)
                neg_span.append([s, s+span])

    return neg_masks, neg_labels

def extract_entity(words, tags, parent_dict={}):
    entity_list, span_list = [], []
    start_index = 0
    end_index = 0

    for i in range(len(tags)):
        if tags[i][0] == 'B':
            start_index = i
            entity_type = tags[i][2:]
        if i > 0:
            if tags[i][0] == 'O' and tags[i-1][0] != 'O':
                end_index = i
                entity_name = words[start_index:end_index]
                entity_list.append((entity_name, entity_type))
                span_list.append([start_index, end_index])
            elif i == len(tags) - 1 and tags[i][0] != 'O':
                end_index = i+1
                entity_name = words[start_index:end_index]
                entity_type = tags[i][2:]
                entity_list.append((entity_name, entity_type))
                span_list.append([start_index, end_index])

    pos_masks, pos_labels = extract_pos(words, entity_list)
    # if len(parent_dict) > 0:
    #     neg_masks, neg_labels = extract_neg_from_hierarchy(words, entity_list, parent_dict)
    # neg_masks, neg_labels = extract_neg(words, span_list)

    # print(f"Positive: {len(pos_masks)}")
    # orig_len = len(words)
    # for i in range(len(pos_masks)):
    #     print(pos_masks[i][orig_len:])
    #     print(pos_labels[i][orig_len:])

    # print(f"Negative: {len(neg_masks)}")
    # for i in range(len(neg_masks)):
    #     print(neg_masks[i][orig_len:])
    #     print(neg_labels[i][orig_len:])

    return entity_list, [' '.join(s) for s in pos_masks], [' '.join(s) for s in pos_labels]#, [' '.join(s) for s in neg_masks], [' '.join(s) for s in neg_labels]


def traverse_hierarchy(child_dict):
    pos_masks = []
    pos_labels = []
    neg_masks = []
    neg_labels = []
    for k in child_dict:
        parent_entity = child_dict['name']
        if k == 'name':
            continue
        child_entity = child_dict[k]['name'] 

        if parent_entity != 'root':
            pos_mask = 'The term'.split(' ') + [child_entity] + 'is a type of <mask> .'.split(' ')
            pos_label = 'The term'.split(' ') + [child_entity] + 'is a type of'.split(' ') + [parent_entity] + ['.']  
            pos_masks.append(pos_mask)
            pos_labels.append(pos_label)

            # pos_mask = 'The term'.split(' ') + '<mask> is a type of'.split(' ') + [parent_entity] + ['.']
            # pos_label = 'The term'.split(' ') + [child_entity] + 'is a type of'.split(' ') + [parent_entity] + ['.']  
            # pos_masks.append(pos_mask)
            # pos_labels.append(pos_label)
        
        if len(child_dict) > 1:
            siblings = [child_dict[child]['name'] for child in child_dict if (child != 'name' and child != child_entity) ]

        if len(siblings) > 0 and len(child_dict[k]) > 1:
            children = [child_dict[k][child]['name'] for child in child_dict[k] if (child != 'name')]
            for sibling in siblings:
                for c in children:
                    neg_mask = 'The term'.split(' ') + [c] + 'is a type of <mask> .'.split(' ')
                    neg_label = 'The term'.split(' ') + [c] + 'is a type of'.split(' ') + [sibling] + ['.']  
                    neg_masks.append(neg_mask)
                    neg_labels.append(neg_label)

        if len(siblings) > 0:
            for sibling in siblings:
                neg_mask = 'The term'.split(' ') + [child_entity] + 'is a type of <mask> .'.split(' ')
                neg_label = 'The term'.split(' ') + [child_entity] + 'is a type of'.split(' ') + [sibling] + ['.']  
                neg_masks.append(neg_mask)
                neg_labels.append(neg_label)

        new_pos_masks, new_pos_labels, new_neg_masks, new_neg_labels = traverse_hierarchy(child_dict[k])
        pos_masks.extend(new_pos_masks)
        pos_labels.extend(new_pos_labels)
        neg_masks.extend(new_neg_masks)
        neg_labels.extend(new_neg_labels)

    return pos_masks, pos_labels, neg_masks, neg_labels

def extract_relations_from_hierarchy(filename):
    child_dict, parent_dict, level_dict = extract_hierarchy(filename)
    pos_masks, pos_labels, neg_masks, neg_labels = traverse_hierarchy(child_dict)
    return [' '.join(s) for s in pos_masks], [' '.join(s) for s in pos_labels], \
        [' '.join(s) for s in neg_masks], [' '.join(s) for s in neg_labels]
