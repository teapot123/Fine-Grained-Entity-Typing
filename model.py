import os
import torch
print("let's use ", torch.cuda.device_count(), "GPUs!")
import transformers
from transformers import RobertaPreTrainedModel, RobertaTokenizer, RobertaForMaskedLM, RobertaModel, RobertaConfig
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn import KLDivLoss
from transformers.models.roberta.modeling_roberta import RobertaLMHead
from transformers.modeling_outputs import MaskedLMOutput
from torch.autograd import Function

class GradMultiply(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * ctx.lambd), None


def grad_multiply(x, lambd=1):
    return GradMultiply.apply(x, lambd)

class PromptNER(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, label_num):
        super().__init__(config)

        if config.is_decoder:
            print(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.label_project = nn.Linear(config.vocab_size, label_num)
        self.vocab_size = config.vocab_size

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()

    def init_project(self, node_id_list, output_num, new_instances=None):
        with torch.no_grad():
            for i in range(len(node_id_list)):
                self.label_project.weight[i] = torch.full((self.vocab_size,), \
                    1 / output_num)
            for i in range(len(node_id_list)):
                node_id = node_id_list[i]
                for j in range(output_num):
                    if i == j:
                        self.label_project.weight[j][node_id] = 1.0
                    else:
                        self.label_project.weight[j][node_id] = 0.0
            if new_instances is not None:
                for i in range(len(node_id_list)):
                    node = node_id_list[i]
                    if node not in new_instances:
                        continue
                    for k, v in new_instances[node]:
                        self.label_project.weight[i][k] += v

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        masked_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        relevant_labels=None,
        neg_sample = 0,
        lambd = 1,
        project = True,
        fix_encoder = False,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if fix_encoder:
            with torch.no_grad():
                outputs = self.roberta(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                sequence_output = outputs[0]
                prediction_scores = self.lm_head(sequence_output)

                batch_size,max_len,vocab_size = prediction_scores.shape
                score = prediction_scores.softmax(dim=-1)
                # masked_label = torch.gather(labels, 1, masked_ids)
                masked_label = labels
            
                masked_ids = masked_ids.repeat(1, 1, vocab_size).reshape(batch_size, 1, vocab_size)
                score = torch.gather(score, 1, masked_ids) # batch x 1 x vocab_size
        else:
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]
            prediction_scores = self.lm_head(sequence_output)

            batch_size,max_len,vocab_size = prediction_scores.shape
            score = prediction_scores.softmax(dim=-1)
            # masked_label = torch.gather(labels, 1, masked_ids)
            masked_label = labels
        
            masked_ids = masked_ids.repeat(1, 1, vocab_size).reshape(batch_size, 1, vocab_size)
            score = torch.gather(score, 1, masked_ids) # batch x 1 x vocab_size

        if labels is not None:
            # loss_fct = CrossEntropyLoss()
            loss_fct = KLDivLoss(reduction='batchmean')
            if project:
                score = score.view(batch_size, vocab_size)
                score = grad_multiply(score, lambd=lambd)
                project_score = self.label_project(score)
                log_score_prob = torch.log(F.softmax(project_score, dim=-1))
                if len(masked_label.shape) == 1 or (len(masked_label.shape) == 2 and masked_label.shape[1] == 1):
                    masked_label = masked_label.view(-1)
                    masked_label = F.one_hot(masked_label, num_classes=log_score_prob.shape[1]).to(torch.float32)
                masked_lm_loss = loss_fct(log_score_prob, masked_label)
                score = F.softmax(project_score.view(batch_size, 1, -1), dim=-1)
            elif relevant_labels is not None:
                relevant_labels = relevant_labels.repeat(batch_size, 1, 1)
                score = torch.gather(score, 2, relevant_labels)
                if len(masked_label.shape) == 1 or (len(masked_label.shape) == 2 and masked_label.shape[1] == 1):
                    masked_label = masked_label.view(-1)
                    masked_label = F.one_hot(masked_label, num_classes=score.shape[2]).to(torch.float32)
                if neg_sample > 0:
                    relevant_score = score.view(batch_size, -1)
                    masked_label = masked_label.view(-1, 1)
                    gold_score = torch.gather(relevant_score, 1, masked_label)
                    total_label_num = relevant_score.shape[1]
                    all_label_ids = torch.tensor([y for y in range(total_label_num)]).to(device)
                    false_label_id = torch.tensor([[y for y in all_label_ids if y not in x] for x in masked_label])
                    false_label_id = false_label_id.to(device)
                    # print(false_label_id)
                    # exit(1)
                    remain_score = torch.gather(relevant_score, 1, false_label_id)
                    false_top_index = torch.topk(remain_score, neg_sample)
                    false_top_score = torch.gather(remain_score, 1, false_top_index.indices)
                    cat_score = torch.cat((gold_score, false_top_score), dim=1)
                    cat_label = torch.tensor([0]*batch_size).to(device)
                    masked_lm_loss = loss_fct(cat_score.view(batch_size, -1), cat_label.view(-1))
                else:
                    log_score_prob = torch.log(F.softmax(score, dim=-1))
                    masked_lm_loss = loss_fct(log_score_prob.view(batch_size, -1), masked_label)
            else:
                if len(masked_label.shape) == 1 or (len(masked_label.shape) == 2 and masked_label.shape[1] == 1):
                    masked_label = masked_label.view(-1)
                    masked_label = F.one_hot(masked_label, num_classes=self.config.vocab_size).to(torch.float32)
                log_score_prob = torch.log(F.softmax(score, dim=-1))
                masked_lm_loss = loss_fct(log_score_prob.view(-1, self.config.vocab_size), masked_label)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output


        return MaskedLMOutput(loss=masked_lm_loss,
            logits=score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def predict(
        self,
        input_ids=None,
        masked_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        with torch.no_grad():
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                return_dict=return_dict,
            )
            sequence_output = outputs[0]
            prediction_scores = self.lm_head(sequence_output)

            batch_size,max_len,vocab_size = prediction_scores.shape
            score = prediction_scores.softmax(dim=-1)
       
            masked_ids = masked_ids.repeat(1, 1, vocab_size).reshape(batch_size, 1, vocab_size)
            score = torch.gather(score, 1, masked_ids) # batch x 1 x vocab_size
            return score
