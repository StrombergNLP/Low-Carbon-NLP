from transformers import TrainerCallback
from carbontracker.tracker import CarbonTracker

class CarbonTrackerCallback(TrainerCallback):
    def __init__(self, max_epochs):
        super().__init__()
        self.tracker = CarbonTracker(epochs=max_epochs)

    def on_epoch_begin(self, **kwargs):
        self.tracker.epoch_start()

    def on_epoch_end(self, **kwargs):
        self.tracker.epoch_end()

    def on_train_end(self, **kwargs):
        self.tracker.stop()
