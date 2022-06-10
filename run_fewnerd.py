import os
from tkinter import FALSE
os.environ["CUDA_VISIBLE_DEVICES"]="6"
import torch
from transformers import RobertaTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from model import *
from data import *

tokenizer = RobertaTokenizer.from_pretrained('/shared/data2/jiaxinh3/FINER/pre-trained')
dataset = 'fewnerd'
bz_list = [8]
lr_list = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3]
lambd_list = [1e-7]#, 1e-5, 1e-4, 1e-3]


def print_sample(train_sent, sent_label, sub_label_ids, logits, node_id_list):
    # train_sent batch x seq_len
    # sent_label batch x seq_len 
    # score batch x 1 x vocab_size
    printed_False = False
    printed_True = False
    for i in range(len(train_sent)):
        true_label = node_id_list[sub_label_ids[i]].item()
        pred_label = node_id_list[torch.argmax(logits[i][0])].item()
        if printed_True and printed_False:
            break
        if printed_True and ( pred_label == true_label ):
            continue
        if printed_False and ( pred_label != true_label ):
            continue
        if pred_label == true_label:
            printed_True = True
        else:
            printed_False = True
        sub_score = logits[i][0].softmax(dim=-1) # vocab_size
        values, predictions = sub_score.topk(5)

        orig_text = tokenizer.decode(train_sent[i]).split('</s>')[0]
        label_name = tokenizer.decode(sent_label[i]).split('</s>')[0]
        print(f"Text: {orig_text}")
        print(f"Label: {label_name}")
        
        for i, (v, p) in enumerate(zip(values.tolist(), predictions.tolist())):
            p0 = node_id_list[p]
            print({"score": v, "token": p, "token_str": tokenizer.decode(p0)})


def prompt_new_instance(unmasker, new_instance_dict, entity_lists, it, type_score,\
    node_id_list, sample_num=3, epsilon=0.9):
    words_1, entity_lists_1 = entity_lists
    words = words_1[it[0]:it[-1]+1]
    entity_list = entity_lists_1[it[0]:it[-1]+1]
    train_sent, sent_att, sent_mask, sent_id = [], [], [], []
    max_seq_len = 128

    for i in tqdm(range(len(entity_list)), desc="finding new instances"):
        entity_name, entity_type = entity_list[i][0]
        _, pred_type = type_score[i][0].topk(1)
        word_len = len(entity_name)
        for samp in range(sample_num):
            new_instance_phrase = ['<mask>'] * word_len
            
            for k in range(len(entity_name)):
                pos_mask = words[i] + entity_name + ', as well as'.split(' ') +\
                    new_instance_phrase + \
                    ', is a'.split(' ') + [tokenizer.decode(pred_type).strip()] + ['.']
                pos_mask = ' '.join(pos_mask)
                ls = unmasker(pos_mask, top_k=sample_num)
                max_score = 0.0
                max_ind = 0
                max_word = ''
                if isinstance(ls[0], list):
                    for j, l in enumerate(ls):
                        if l[samp]['score'] > max_score:
                            max_score = l[samp]['score']
                            max_ind = j
                            max_word = l[samp]['token_str'].strip()
                else:
                    max_score = ls[samp]['score']
                    max_ind = 0
                    max_word = ls[samp]['token_str'].strip()
                mask_j = 0
                point = 0
                for point in range(len(new_instance_phrase)):
                    if new_instance_phrase[point] != '<mask>':
                        continue
                    if mask_j == max_ind:
                        new_instance_phrase[point] = max_word
                        break
                    mask_j += 1

            new_entity_name = ' '.join(entity_name)
            if new_entity_name not in new_instance_dict[entity_type]:
                new_instance_dict[entity_type][new_entity_name] = []
            new_instance_dict[entity_type][new_entity_name].append(\
                tokenizer.decode(pred_type).strip()+':'\
                +' '.join(new_instance_phrase))

            if all(j in string.punctuation for j in new_instance_phrase):
                continue
            pos_mask = words[i] + new_instance_phrase + \
                    'is a <mask> .'.split(' ')
            pos_mask = ' '.join(pos_mask)     
            pos_mask = tokenizer(pos_mask, return_tensors='pt', max_length=max_seq_len, padding='max_length', truncation=True)  
            masked_id = torch.nonzero(pos_mask["input_ids"] == tokenizer.mask_token_id, as_tuple=False)[:, 1].unsqueeze(1)
            if len(masked_id) != len(pos_mask["input_ids"]):
                continue
            train_sent.append(pos_mask["input_ids"])
            sent_att.append(pos_mask["attention_mask"])
            sent_mask.append(masked_id)
            label_id = (node_id_list == tokenizer.encode(' '+entity_type)[1]).nonzero(as_tuple=True)[0][0]
            smooth_id = epsilon * F.one_hot(label_id, num_classes=len(node_id_list))
            smooth_id += (1-epsilon) * torch.ones_like(smooth_id) / len(node_id_list)
            sent_id.append(smooth_id.unsqueeze(0))


    train_sent = torch.cat(train_sent, dim=0)
    sent_att = torch.cat(sent_att, dim=0)
    sent_mask = torch.cat(sent_mask, dim=0)
    sent_id = torch.cat(sent_id, dim=0)
    return new_instance_dict, [train_sent, sent_att, sent_mask, sent_id]


def add_keywords(node_id_list, keywords_list):
    x = model.label_project.weight.data.cpu().numpy()
    if keywords_list == []:
        keywords_list = [[] for node in node_id_list]
    l = len(keywords_list[0])
    all_keywords = []
    for i in range(len(keywords_list)):
        all_keywords.extend(keywords_list[i])
 
    for i in range(len(node_id_list)):
        ind = np.argsort(-x[i])[:l+1]
        for j in range(l+1):
            if ind[j] not in all_keywords:
                keywords_list[i].append(ind[j])
                all_keywords.append(ind[j])
                break
    
    # for i in range(len(keywords_list)):
    #     keywords = ' '.join([tokenizer.decode(w) for w in keywords_list[i]])
    #     print(tokenizer.decode(node_id_list[i])+': '+keywords)

    return keywords_list


def cal_disc_loss(node_id_list, keywords_list):
    x = model.label_project.weight.data.cpu().numpy()
    all_keywords = []
    keyword_class = []
    for i in range(len(keywords_list)):
        all_keywords.extend(keywords_list[i])
        keyword_class.extend([i for j in keywords_list[i]])
    

    keyword_class = torch.tensor(keyword_class).to(device)
    all_keywords = torch.tensor(all_keywords).to(device)
    all_keywords = all_keywords.view(-1).repeat(len(node_id_list), 1)
    score = torch.gather(model.label_project.weight, 1, all_keywords).T

    loss_fct = CrossEntropyLoss()
    disc_loss = loss_fct(score, keyword_class)
    return disc_loss
    

def match_results(logits, sub_test_mask, sub_label_ids, node_id_list,
    total_pred, total_gold, total_crct):
    # masked_label = torch.gather(sub_test_label, 1, sub_test_mask)  #masked_label[i][0]
    if word2label is not None:
        word_ids = torch.tensor(list(word2label.keys())).to(device)   
    # logits.shape: batch x 1 x vocab_size
    for i in range(len(logits)):
        true_label = node_id_list[sub_label_ids[i]].item()
        total_crct.append(true_label)
        pred_label = node_id_list[torch.argmax(logits[i][0])].item()
        total_pred.append(pred_label)
    return total_pred, total_gold, total_crct


def loose_f1_score(total_crct, total_pred, reverse_train_types):
    new_total_crct = [reverse_train_types[x] if x in reverse_train_types else [] for x in total_crct]
    new_total_pred = [reverse_train_types[x] if x in reverse_train_types else [] for x in total_pred]
    loose_micro_pred, loose_micro_crct, loose_micro_gold = 0.0, 0.0, 0.0
    loose_macro_p, loose_macro_r = 0.0, 0.0
    for i in range(len(new_total_crct)):
        intersection = [x for x in new_total_crct[i] if x in new_total_pred[i]]
        if len(new_total_pred[i]) > 0:
            loose_macro_p += len(intersection) / len(new_total_pred[i])
        loose_macro_r += len(intersection) / len(new_total_crct[i])
        loose_micro_pred += len(new_total_pred[i])
        loose_micro_crct += len(intersection)
        loose_micro_gold += len(new_total_crct[i])
    loose_micro_p = loose_micro_crct / loose_micro_pred
    loose_micro_r = loose_micro_crct / loose_micro_gold
    loose_micro_f1 = 2 * loose_micro_p * loose_micro_r/ (loose_micro_p + loose_micro_r)
    loose_macro_f1 = 2 * loose_macro_p * loose_macro_r/ (loose_macro_p + loose_macro_r) / len(new_total_crct)
    return loose_micro_f1, loose_macro_f1


def train_func(epoch, N_EPOCHS, pos_data, neg_data, random_iter, batch_size, word2label, neg_sample=0,\
    fix_encoder=False, lambd=1, project=False, keywords_list=[], add_new_instance=False, \
    sample_num=3, temporal_ensemble=False, ensem_score=None, momentum=0.6, temp_ensem_weight=10,\
    epsilon=0.9):
    
    epoch_loss = 0
    train_pos_loss = 0
    train_neg_loss = 0
    total_pred, total_gold, total_crct = [], [], []
    new_train_sent, new_sent_att, new_sent_mask, new_sent_id = [], [], [], []
    new_ensem_scores = []
    training_ratio = epoch / N_EPOCHS
    train_sent, sent_att, sent_mask, sent_label, label_ids, node_id_list, reverse_train_types,\
        new_instances, entity_list = pos_data
    if neg_data is not None:
        neg_sent, neg_att, neg_mask, neg_label = neg_data
    
    if add_new_instance:
        model.eval()
        new_instance_dict = defaultdict(dict)
        predict_batch_size = 128
        step_num = int((len(train_sent) - 1) / predict_batch_size) + 1
        for i in range(step_num):
            it = list(range(i * predict_batch_size, min((i+1) * predict_batch_size, len(train_sent) )))
            if len(it) == 0:
                continue
            
            sub_train_sent, sub_sent_att, sub_sent_mask, sub_sent_label, sub_label_ids = \
                train_sent[it], sent_att[it], sent_mask[it], sent_label[it], label_ids[it]
            sub_train_sent, sub_sent_att, sub_sent_mask, sub_sent_label, sub_label_ids = \
                sub_train_sent.to(device), sub_sent_att.to(device), sub_sent_mask.to(device),\
                sub_sent_label.to(device), sub_label_ids.to(device)
            type_score = model.predict(input_ids=sub_train_sent, masked_ids=sub_sent_mask,
                attention_mask=sub_sent_att)
            new_instance_dict, instance_data = prompt_new_instance(unmasker, new_instance_dict, \
                entity_list, it, type_score, node_id_list, sample_num=sample_num, epsilon=epsilon)
            new_train_sent.append(instance_data[0])
            new_sent_att.append(instance_data[1])
            new_sent_mask.append(instance_data[2])
            new_sent_id.append(instance_data[3])
        new_train_sent = torch.cat(new_train_sent, dim=0)
        new_sent_att = torch.cat(new_sent_att, dim=0)
        new_sent_mask = torch.cat(new_sent_mask, dim=0)
        new_sent_id = torch.cat(new_sent_id, dim=0)

    if temporal_ensemble and epoch > 0:
        model.eval()
        predict_batch_size = 128
        step_num = int((len(train_sent) - 1) / predict_batch_size) + 1
        with torch.no_grad():
            for i in range(step_num):
                it = list(range(i * predict_batch_size, min((i+1) * predict_batch_size, len(train_sent) )))
                
                sub_train_sent, sub_sent_att, sub_sent_mask, sub_sent_label, sub_label_ids = \
                    train_sent[it], sent_att[it], sent_mask[it], sent_label[it], label_ids[it]
                sub_train_sent, sub_sent_att, sub_sent_mask, sub_sent_label, sub_label_ids = \
                    sub_train_sent.to(device), sub_sent_att.to(device), sub_sent_mask.to(device),\
                    sub_sent_label.to(device), sub_label_ids.to(device)
                tmp_output = model(input_ids=sub_train_sent, masked_ids=sub_sent_mask, \
                    attention_mask=sub_sent_att, labels=sub_label_ids, relevant_labels=node_id_list,\
                    neg_sample=neg_sample, fix_encoder=fix_encoder, lambd=lambd, project=project)
                last_epoch_score = tmp_output.logits.view(-1, len(node_id_list))
                if epoch == 1:
                    ensem_score.append((1-momentum) * last_epoch_score)
                    new_ensem_score = ensem_score[-1] / (1 - momentum ** epoch)
                else:
                    ensem_score[it] = (1-momentum)* last_epoch_score + momentum * ensem_score[it]
                    new_ensem_score = ensem_score[it] / (1 - momentum ** epoch)
                new_ensem_scores.append(new_ensem_score)
            new_ensem_scores = torch.cat(new_ensem_scores, dim=0)
            if epoch == 1:
                ensem_score = torch.cat(ensem_score, dim=0)
        
    model.train()
    node_id_list = node_id_list.to(device)
    step_num = int((len(random_iter) - 1) / batch_size) + 1
    for i in tqdm(range(step_num), desc="training steps"):
        it = random_iter[i * batch_size : min((i+1) * batch_size, len(random_iter) )]
        optimizer.zero_grad()

        sub_train_sent, sub_sent_att, sub_sent_mask, sub_sent_label, sub_label_ids = \
            train_sent[it], sent_att[it], sent_mask[it], sent_label[it], label_ids[it]
        sub_train_sent, sub_sent_att, sub_sent_mask, sub_sent_label, sub_label_ids = \
            sub_train_sent.to(device), sub_sent_att.to(device), sub_sent_mask.to(device),\
            sub_sent_label.to(device), sub_label_ids.to(device)
        pos_output = model(input_ids=sub_train_sent, masked_ids=sub_sent_mask,
            attention_mask=sub_sent_att, labels=sub_label_ids, relevant_labels=node_id_list,\
            neg_sample=neg_sample, fix_encoder=fix_encoder, lambd=lambd, project=project)
        pos_loss = pos_output.loss
        loss = pos_loss

        if temporal_ensemble and epoch > 0:
            temp_ensem_label = new_ensem_scores[it].to(device)
            temp_ensem_output = model(input_ids=sub_train_sent, masked_ids=sub_sent_mask, \
                attention_mask=sub_sent_att, labels=temp_ensem_label, relevant_labels=node_id_list,\
                neg_sample=neg_sample, fix_encoder=fix_encoder, lambd=lambd, project=project)
            temp_ensem_loss = temp_ensem_output.loss
            ramp_up = torch.exp(-5*(1-torch.tensor(training_ratio))**2)
            loss += temp_ensem_weight * ramp_up * temp_ensem_loss

        if add_new_instance:
            new_it = random_iter[i * batch_size : min((i+1) * batch_size, len(new_train_sent) )]
            new_it = [x for x in new_it if x < len(new_train_sent)]
            if len(new_it) > 0:
                sub_train_sent1, sub_sent_att1, sub_sent_mask1, sub_label_ids1 = \
                    new_train_sent[new_it], new_sent_att[new_it], new_sent_mask[new_it], new_sent_id[new_it]
                sub_train_sent1, sub_sent_att1, sub_sent_mask1, sub_label_ids1 = \
                    sub_train_sent1.to(device), sub_sent_att1.to(device), sub_sent_mask1.to(device),\
                    sub_label_ids1.to(device)
                new_instance_output = model(input_ids=sub_train_sent1, masked_ids=sub_sent_mask1,
                    attention_mask=sub_sent_att1, labels=sub_label_ids1, relevant_labels=node_id_list,\
                    neg_sample=neg_sample, fix_encoder=fix_encoder, lambd=lambd, project=project)
                new_instance_loss = new_instance_output.loss
                loss += new_instance_loss/2/sample_num
        
        loss.backward()
        
        epoch_loss += loss.item()
        train_pos_loss += pos_loss.item()

        if keywords_list != []:
            disc_loss = cal_disc_loss(node_id_list, keywords_list)
            disc_loss.backward()
            epoch_loss += disc_loss.item()

        optimizer.step()
        scheduler.step()

        total_pred, total_gold, total_crct = match_results(
            pos_output.logits, sub_sent_mask, sub_label_ids, node_id_list,
            total_pred, total_gold, total_crct)

        # if i % 5 == 0:
        #     print_sample(sub_train_sent, sub_sent_label, pos_output.logits, node_id_list)

    if project:
        keywords_list = add_keywords(node_id_list, keywords_list)
    
    acc = f1_score(total_crct, total_pred, average='micro')
    train_micro_f1, train_macro_f1 = loose_f1_score(total_crct, total_pred, reverse_train_types)

    return epoch_loss / len(random_iter), acc, train_macro_f1, train_micro_f1, \
        total_crct, total_pred, keywords_list, ensem_score


def test_func(pos_data, batch_size, word2label, project=False):
    model.eval()
    epoch_loss = 0
    val_loss = 0
    total_pred, total_gold, total_crct = [], [], []
    test_sent, test_att, test_mask, test_label, label_ids, node_id_list, reverse_train_types,\
        new_instances, entity_list = pos_data
    node_id_list = node_id_list.to(device)
    step_num = int((len(test_sent) - 1) / batch_size) + 1

    for i in tqdm(range(step_num), desc="testing steps"):
        it = list(range(i * batch_size, min((i+1) * batch_size, len(test_sent) )))

        sub_test_sent, sub_test_att, sub_test_mask, sub_test_label, sub_label_ids = \
            test_sent[it], test_att[it], test_mask[it], test_label[it], label_ids[it]
        sub_test_sent, sub_test_att, sub_test_mask, sub_test_label, sub_label_ids = \
            sub_test_sent.to(device), sub_test_att.to(device), sub_test_mask.to(device), \
            sub_test_label.to(device), sub_label_ids.to(device)

        with torch.no_grad():
            output = model(input_ids=sub_test_sent, masked_ids=sub_test_mask,
                attention_mask=sub_test_att, labels=sub_label_ids, relevant_labels=node_id_list, \
                project=project)
        loss = output.loss
        val_loss += loss.item()

        total_pred, total_gold, total_crct = match_results(
                output.logits, sub_test_mask, sub_label_ids, node_id_list,
                total_pred, total_gold, total_crct)

        # if i % 5 == 0:
        #     print_sample(sub_test_sent, sub_test_label, sub_label_ids, output.logits, node_id_list)
    
    acc = f1_score(total_crct, total_pred, average='micro')
    test_micro_f1, test_macro_f1= loose_f1_score(total_crct, total_pred, reverse_train_types)

    return epoch_loss / len(test_sent), acc, test_macro_f1, test_micro_f1, total_crct, total_pred


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='main',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', default=5e-5,type=float)
    parser.add_argument('--epochs', default=20,type=int)
    parser.add_argument('--lambd', default=1e-7,type=float)
    parser.add_argument('--new_inst_num', default=3,type=int)
    parser.add_argument('--shot_num', default=5,type=int)
    parser.add_argument('--add_new_inst', default=0,type=int)
    parser.add_argument('--momentum', default=0.6,type=float)
    parser.add_argument('--epsilon', default=0.9,type=float)
    parser.add_argument('--temp_ensem_weight', default=1.0,type=float)

    args = parser.parse_args()
    print(args)
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.epochs
    lr = args.lr
    lambd = args.lambd
    new_inst_num = args.new_inst_num
    momentum = args.momentum
    temp_ensem_weight = args.temp_ensem_weight
    shot_num = args.shot_num
    epsilon = args.epsilon
    if args.add_new_inst > 0:
        add_new_inst = True
    else:
        add_new_inst = False
    project = True

    with open(os.path.join('results', dataset, 'vary_epsilon.txt'), 'a') as fout:
        fout.write(f'args: {args}\n')
        train_acc_average = 0
        train_micro_f1_average = 0
        train_macro_f1_average = 0
        test_acc_average = 0
        test_micro_f1_average = 0
        test_macro_f1_average = 0
        for i in range(1,2):
            train_file = os.path.join('data', dataset, dataset+'_train_new_few_shot_5_'+str(i)+'.json')
            # train_file = os.path.join('data', dataset, 'train_'+str(shot_num)+'_'+str(i)+'.json')
            # train_file = os.path.join('data', dataset, 'supervised.json')
            test_file = os.path.join('data', dataset, dataset+'_test_new_5_'+str(i)+'.json')
            # test_file = os.path.join('data', dataset, dataset+'_test.json')
            new_hier = dataset+'_hier.txt'
            old_hier = os.path.join('data', 'ontology', dataset+'_types.txt')
            # old_hier = os.path.join('data', 'ontology', dataset+'_types_old.txt')

            # train_type_count = read_types(test_file)
            train_type_count_1 = read_types(train_file)
            train_type_count_2 = read_types(test_file)
            train_type_count = [value for value in train_type_count_1 if value in train_type_count_2]
            pos_data = read_file_and_hier(train_file, new_hier, old_hier, train_type_count, use_node_list=True,\
                extract_new_instance=False)
            word2label = None
            node_id_list = pos_data[5]
            print(len(train_type_count))
            print(f"Positive samples: {len(pos_data[0])}")
            new_instances = pos_data[7]
            test_data = read_file_and_hier(test_file, new_hier, old_hier, train_type_count, use_node_list=True)
            for j in range(1):
                model = PromptNER.from_pretrained('/shared/data2/jiaxinh3/FINER/pre-trained', label_num=len(node_id_list))
                model.init_project(node_id_list, output_num=len(node_id_list), new_instances=new_instances)
                device = torch.device("cuda")
                model.to(device)
                param_optimizer = list(model.named_parameters())

                no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in param_optimizer
                                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                    {'params': [p for n, p in param_optimizer
                                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

                num_training_steps = N_EPOCHS * len(pos_data[0]) / BATCH_SIZE
                num_warmup_steps = int(0.1 * num_training_steps)
                print(f"num training steps: {num_training_steps}")
                print(f"num warmup steps: {num_warmup_steps}")
                optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=True)
                # optimizer = Adam(optimizer_grouped_parameters, lr=args.lr, betas=(0.9,0.98), eps=1e-6)
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
                
                keywords_list = []
                ensem_score = []
                for epoch in range(N_EPOCHS):
                    random_iter = np.arange(len(pos_data[0]))
                    np.random.shuffle(random_iter)
                    if epoch < N_EPOCHS/2:
                        train_loss, train_acc, train_macro_f1, train_micro_f1, train_pos_loss, train_neg_loss,\
                            keywords_list, ensem_score = train_func(epoch, N_EPOCHS, pos_data, None, \
                            random_iter, BATCH_SIZE, word2label=None, fix_encoder = False, lambd = lambd, project=project,\
                            keywords_list = keywords_list, add_new_instance=False, temporal_ensemble=False, \
                            ensem_score=ensem_score, momentum=momentum, temp_ensem_weight=temp_ensem_weight)
                        test_loss, test_acc, test_macro_f1, test_micro_f1, total_crct, total_pred = test_func(\
                            test_data, 128, word2label=None, project=project)
                    else:
                        train_loss, train_acc, train_macro_f1, train_micro_f1, train_pos_loss, train_neg_loss,\
                            keywords_list, ensem_score = train_func(epoch, N_EPOCHS, pos_data, None, \
                            random_iter, BATCH_SIZE, word2label=None, fix_encoder = False, lambd = lambd, project=project,\
                            keywords_list = keywords_list, add_new_instance=add_new_inst, sample_num=5,\
                            temporal_ensemble=False, ensem_score=ensem_score, momentum=momentum, temp_ensem_weight=temp_ensem_weight,\
                            epsilon=epsilon)
                        test_loss, test_acc, test_macro_f1, test_micro_f1, total_crct, total_pred = test_func(\
                            test_data, 128, word2label=None, project=project)
                    fout.write(f"Train:\t{train_acc * 100:.2f}\t{train_micro_f1 * 100:.2f}\t{train_macro_f1 * 100:.2f}\tTest:\t{test_acc * 100:.2f}\t{test_micro_f1 * 100:.2f}\t{test_macro_f1 * 100:.2f}\n")
                
                train_acc_average += train_acc
                train_micro_f1_average += train_micro_f1
                train_macro_f1_average += train_macro_f1

                test_loss, test_acc, test_macro_f1, test_micro_f1, total_crct, total_pred = test_func(\
                    test_data, 128, word2label=None, project=project)

                test_acc_average += test_acc
                test_micro_f1_average += test_micro_f1
                test_macro_f1_average += test_macro_f1

        # train_acc_average /= 5
        # train_micro_f1_average /= 5
        # train_macro_f1_average /= 5
        # test_acc_average /= 5
        # test_micro_f1_average /= 5
        # test_macro_f1_average /= 5
        fout.write(f"Train:\t{train_acc_average * 100:.2f}\t{train_micro_f1_average * 100:.2f}\t{train_macro_f1_average * 100:.2f}\tTest:\t{test_acc_average * 100:.2f}\t{test_micro_f1_average * 100:.2f}\t{test_macro_f1_average * 100:.2f}\n")
        

