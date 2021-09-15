import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput


class FormationPredictor(nn.Module):
    def __init__(self, hidden_size=768, formation_num=18):
        super(FormationPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.classifier = nn.Linear(hidden_size, formation_num)
    def forward(self, context_embedding):
        hidden = self.mlp(context_embedding)
        hidden_pooled = torch.mean(hidden, dim=1) # mean-pooling over the seq-len dim
 
        return self.classifier(hidden_pooled)


class FormationMTLBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_formation=18, formation_hidden_size=768, formation_cls_weight=0.5):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        if formation_hidden_size != config.hidden_size:
            formation_hidden_size = config.hidden_size
        self.formation_embedding = nn.Embedding(num_formation, formation_hidden_size)
        self.formation_cls_weight = formation_cls_weight  # weight for adding formation classification loss
        self.formation_predictor = FormationPredictor(config.hidden_size, num_formation)
        self.num_formation = num_formation
        self.init_weights()
        self.mode = 'wsd'

    def set_mode(self, mode='wsd'):
        self.mode = mode

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            formation=None,
            # input for context
            context_input_ids=None,
            context_token_type_ids=None,
            context_position_ids=None,
            # original output control
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        context_embedding = self.bert.embeddings(input_ids=context_input_ids,
                                                 token_type_ids=context_token_type_ids,
                                                 position_ids=context_position_ids)
        #  utilize context embedding for inferring the formation type
        formation_logits = self.formation_predictor(context_embedding)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        # directly add formation here for now
        if self.training:  # training, using ground-truth for foramtion
            formation_embedding = self.formation_embedding(formation)
        else:
            predicted_formation = torch.argmax(formation_logits, dim=-1)
            formation_embedding = self.formation_embedding(predicted_formation)

        pooled_output = pooled_output + formation_embedding  #
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss += self.formation_cls_weight * loss_fct(formation_logits.view(-1, self.num_formation),
                                                             formation.view(-1))
        if self.mode == "formation":  # predict formation
            logits = formation_logits

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


