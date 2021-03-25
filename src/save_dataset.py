from datasets import load_dataset

dataset = load_dataset('cc_news', script_version='master')

dataset_reduced = dataset['train']['text'][:100000]
del dataset

dataset_reduced.save_to_disk('cc_news_reduced')