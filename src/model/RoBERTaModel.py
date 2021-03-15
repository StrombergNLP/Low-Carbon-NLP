from torch import nn
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM

class RoBERTaModel(nn.Module):
    """
    Class for tuning any given model architecture
    """

    def __init__(
            self,
            model_parameters,
            output_hidden_states=True,
            output_logits=True,
            output_attentions=True
        ):

        super().__init__()

        self.config = RobertaConfig(
            vocab_size=model_parameters['vocab_size'],
            hidden_size=model_parameters['hidden_size'] * model_parameters['num_attention_heads'],
            num_hidden_layers=model_parameters['num_hidden_layers'],
            num_attention_heads=model_parameters['num_attention_heads'],
            intermediate_size=model_parameters['intermediate_size'],
            hidden_act=model_parameters['hidden_act'],
            hidden_dropout_prob=model_parameters['hidden_dropout_prob'],
            attention_probs_dropout_prog=model_parameters['attention_probs_dropout_prog'],
            max_position_embeddings=model_parameters['max_position_embeddings'] * 2,
            type_vocab_size=model_parameters['type_vocab_size'],
            initializer_range=model_parameters['initializer_range'],
            layer_norm_eps=model_parameters['layer_norm_eps'],
            gradient_checkpointing=model_parameters['gradient_checkpointing'],
            position_embedding_type=model_parameters['position_embedding_type'],
            use_cache=model_parameters['use_cache'],
            output_hidden_states=output_hidden_states,
            output_logits=output_logits,
            output_attentions=output_attentions
        )
        self.model = RobertaForMaskedLM(config=self.config)

        # self.tune()


    def forward(self, *model_args,**model_kwargs):
        return self.model(*model_args, **model_kwargs)


    def tune(self):
        state_dict = self.model.state_dict()
        # print(state_dict.keys())

