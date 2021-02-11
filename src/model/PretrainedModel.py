from torch import nn
import transformers

class Model(nn.Module):
    """
    Class for tuning any given model architecture
    """

    def __init__(
            self,
            model_name,
            output_hidden_states=True,
            output_logits=True,
            output_attentions=True
        ):

        super().__init__()
        self.config = transformers.AutoConfig.from_pretrained(model_name, output_hidden_states=output_hidden_states, output_logits=output_logits)
        self.model = transformers.AutoModel.from_pretrained(model_name, output_attentions=output_attentions)

        self.tune()


    def forward(self, *model_args,**model_kwargs):
        return self.model(*model_args, **model_kwargs)


    def tune(self):
        state_dict = self.model.state_dict()
        # print(state_dict.keys())

