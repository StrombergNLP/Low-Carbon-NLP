import os

from transformers import TrainerCallback, Trainer, TrainingArguments, TrainerState, TrainerControl, PreTrainedModel

class SaveModelCallback(TrainerCallback):
    def __init__(self, save_path, model_id):
        super().__init__()
        self.save_path = save_path + '/' + 'models/' + 'model_' + str(model_id)

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)


    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: PreTrainedModel, **kwargs):
        epoch = int(state.epoch)
        save_directory = self.save_path + '/' + 'epoch_' + str(epoch)

        if not os.path.exists(save_directory):
            os.mkdir(save_directory)

        print('######################')
        print('MODEL TYPE')
        print(type(model))
        print('######################')
        model.save_pretrained(save_directory=save_directory)

