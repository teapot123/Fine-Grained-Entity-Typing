from tqdm import tqdm
import numpy as np
import string
import torch
import torch.nn.functional as F

def generate_candidate_instance(unmasker, tokenizer, candidate_scores, words, entity_name, new_instance_phrase, pred_type, \
    candidate_num):
    stop_word_list = ['other', 'his', 'her', 'its', 'this', 'it', 'another']
    l = len(new_instance_phrase)
    count = 0
    candidate_scores = {}
    top_k = candidate_num
    while count < candidate_num:
        order_list = np.random.choice(l, size = l, replace=False).tolist()
        new_instance_phrase = ['<mask>'] * l
        ll_score = 0.0
        for k in range(l):
            pos_mask = ' '.join(words) + ' ' + ' '.join(entity_name) + ' , as well as ' +\
                ''.join(new_instance_phrase) + \
                ' , is a ' + tokenizer.decode(pred_type).strip() + ' .'
            # pos_mask = ' '.join(pos_mask)
            ls = unmasker(pos_mask, top_k=top_k)

            if isinstance(ls[0], list):
                filled=np.sum(np.array(order_list[:k])<np.array(order_list[k]))
                ls = ls[order_list[k]-filled]
            score_list = np.array([cand['score'] for cand in ls])

            sample_ind = np.random.choice(top_k, size=1, replace=False, \
                p=score_list/sum(score_list)).tolist()[0]
            sample_word = ls[sample_ind]['token_str']
            ll_score += np.log(ls[sample_ind]['score'])
            new_instance_phrase[order_list[k]] = sample_word
            
        # print(f"{new_instance_phrase}: {ll_score}")
        if any(j.strip().lower() in stop_word_list for j in new_instance_phrase):
            top_k += 5
            continue
        elif all(j in string.punctuation for j in new_instance_phrase):
            top_k += 5
            continue
        else:
            count += 1
        candidate_scores[''.join(new_instance_phrase)] = ll_score

    return candidate_scores


def generate_new_instance(unmasker, tokenizer, new_instance_dict, entity_lists, it, type_score,\
    node_id_list, sample_num=3):
    words_1, entity_lists_1 = entity_lists
    words = words_1[it[0]:it[-1]+1]
    entity_list = entity_lists_1[it[0]:it[-1]+1]
    train_sent, sent_att, sent_mask, sent_id = [], [], [], []
    max_seq_len = 128
    stop_word_list = ['other', 'his', 'her', 'its', 'this', 'it', 'another']
    candidate_ratio = 10
    
    for i in tqdm(range(len(entity_list)), desc="finding new instances"):
        entity_name, entity_type = entity_list[i][0]
        _, pred_type = type_score[i][0].topk(1)
        word_len = len(entity_name)
        sqr_total = word_len * (word_len + 1) / 2
        candidate_scores = {}
        for l in range(1, word_len+1):
            candidate_num = int(sample_num * candidate_ratio / sqr_total * l) + 1
            
            new_instance_phrase = ['<mask>'] * l
            candidate_scores = generate_candidate_instance(unmasker, tokenizer, candidate_scores, words[i], entity_name, \
                new_instance_phrase, pred_type, candidate_num)
            
        new_entity_name = ' '.join(entity_name)
        if new_entity_name not in new_instance_dict[entity_type]:
            new_instance_dict[entity_type][new_entity_name] = []
        # for k,v in candidate_scores.items():
        #     new_instance_dict[entity_type][new_entity_name].append(\
        #         tokenizer.decode(pred_type).strip()+':'\
        #         +k + ' (' + str(v) + ') ')
            
        sorted_candidates = sorted(candidate_scores.items(), key=lambda item: -item[1])
        for k,v in sorted_candidates[:sample_num]:
            # print(f"{k}: {v}")
            new_instance_dict[entity_type][new_entity_name].append(\
                tokenizer.decode(pred_type).strip()+':'\
                +k + ' (' + str(v) + ') ')
            pos_mask = ' '.join(words[i]) + ' ' + k + \
                ' is a <mask> .'
            pos_mask = tokenizer(pos_mask, return_tensors='pt', max_length=max_seq_len, padding='max_length', truncation=True)  
            masked_id = torch.nonzero(pos_mask["input_ids"] == tokenizer.mask_token_id, as_tuple=False)[:, 1].unsqueeze(1)
            if len(masked_id) != len(pos_mask["input_ids"]):
                continue
            train_sent.append(pos_mask["input_ids"])
            sent_att.append(pos_mask["attention_mask"])
            sent_mask.append(masked_id)
            label_id = (node_id_list == tokenizer.encode(' '+entity_type)[1]).nonzero(as_tuple=True)[0][0]
            smooth_id = 0.9 * F.one_hot(label_id, num_classes=len(node_id_list))
            smooth_id += 0.1 * torch.ones_like(smooth_id) / len(node_id_list)
            sent_id.append(smooth_id.unsqueeze(0))

    train_sent = torch.cat(train_sent, dim=0)
    sent_att = torch.cat(sent_att, dim=0)
    sent_mask = torch.cat(sent_mask, dim=0)
    sent_id = torch.cat(sent_id, dim=0)
    return new_instance_dict, [train_sent, sent_att, sent_mask, sent_id]
