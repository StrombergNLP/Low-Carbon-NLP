from transformers import TrainerCallback, Trainer, TrainingArguments, TrainerState, TrainerControl

class PerplexityCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        loss = float(metrics.get('loss'))
        print('Here is the perplexity: {}\n'.format(2.0**loss))

