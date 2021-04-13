import os, time

from carbontracker import parser
from carbontracker.tracker import CarbonTracker


epochs = 10
log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'carbonlogs'))
tracker = CarbonTracker(epochs=epochs, epochs_before_pred=-1, monitor_epochs=-1, verbose=2, log_dir=log_path)


for i in range(epochs):
    tracker.epoch_start()
    time.sleep(2)
    tracker.epoch_end()
    time.sleep(2)
    logs = parser.parse_all_logs(log_dir=log_path)
    print(logslogs[len(logs)-1]['actual'])

tracker.stop()
