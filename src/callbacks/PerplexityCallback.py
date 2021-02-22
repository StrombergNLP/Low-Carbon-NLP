from transformers import TrainerCallback, Trainer, TrainingArguments, TrainerState, TrainerControl

class PerplexityCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metric, **kwargs):
        print(metric)

