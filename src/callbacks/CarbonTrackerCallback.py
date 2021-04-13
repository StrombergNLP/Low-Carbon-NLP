import csv, math

from transformers import TrainerCallback, Trainer, TrainingArguments, TrainerState, TrainerControl
from carbontracker.tracker import CarbonTracker
from carbontracker import parser

# class CarbonTrackerCallback(TrainerCallback):
#     def __init__(self, max_epochs, log_path):
#         super().__init__()
#         self.tracker = CarbonTracker(epochs=max_epochs, epochs_before_pred=-1, monitor_epochs=-1, verbose=2, log_dir=log_path)

#     def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
#         self.tracker.epoch_start()

#     def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
#         self.tracker.epoch_end()

#     def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
#         self.tracker.stop()


class CarbonTrackerCallback(TrainerCallback):
    def __init__(self, max_epochs, log_path, model_id, results_path, params_file_name):
        super().__init__()
        self.max_epochs = max_epochs
        self.log_path = log_path
        self.model_id = model_id
        self.results_path = results_path
        self.params_file_name = params_file_name
        self.energy_consumption = []
        self.loss = []


    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.tracker = CarbonTracker(epochs=1, epochs_before_pred=-1, monitor_epochs=-1, verbose=2, log_dir=self.log_path)
        self.tracker.epoch_start()


    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.tracker.epoch_end()
        self.tracker.stop()
        logs = parser.parse_all_logs(log_dir=self.log_path)
        latest_log = logs[len(logs) - 1]
        self.energy_consumption.append(latest_log['actual']['energy (kWh)'])
        loss = state.log_history[len(state.log_history) - 1]['loss']
        self.loss.append(loss)


    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        epochs = list(range(1, self.max_epochs + 1))
        csv_columns = list(map(str, epochs))
        csv_columns.insert(0, 'id')

        per_epoch_perplexity = {'id': self.model_id}
        per_epoch_consumptions = {'id': self.model_id}
        for i in epochs:
            consumption = sum(self.energy_consumption[:i])
            per_epoch_consumptions[str(i)] = consumption
            per_epoch_perplexity[str(i)] = math.exp(self.loss[i-1])

        perplexity_average = math.exp(sum(self.loss)/3)
        perplexity_final = math.exp(self.loss[-1])

        with open(self.results_path + '/' + self.params_file_name + '_energy_per_epoch.csv', 'a+') as result_file:
            writer = csv.DictWriter(result_file, fieldnames=csv_columns)
            writer.writerow(per_epoch_consumptions)

        csv_columns.append('average_loss_perplexity')
        csv_columns.append('final_perplexity')

        per_epoch_perplexity['average_loss_perplexity'] = perplexity_average
        per_epoch_perplexity['final_perplexity'] = perplexity_final

        with open(self.results_path + '/' + self.params_file_name + '_perplexity_per_epoch.csv', 'a+') as result_file:
            writer = csv.DictWriter(result_file, fieldnames=csv_columns)
            writer.writerow(per_epoch_perplexity)

