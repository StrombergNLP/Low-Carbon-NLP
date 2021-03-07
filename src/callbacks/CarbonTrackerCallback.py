from transformers import TrainerCallback, Trainer, TrainingArguments, TrainerState, TrainerControl
from carbontracker.tracker import CarbonTracker

class CarbonTrackerCallback(TrainerCallback):
    def __init__(self, max_epochs):
        super().__init__()
        self.tracker = CarbonTracker(epochs=max_epochs, epochs_before_pred=-1, monitor_epochs=-1)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.tracker.epoch_start()

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.tracker.epoch_end()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass