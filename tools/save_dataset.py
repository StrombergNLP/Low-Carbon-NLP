from datasets import load_dataset

dataset = load_dataset('cc_news', script_version='master')

dataset_reduced = dataset['train']['text'][:8000]
del dataset

with open('cc_news_reduced_2.txt', 'w+', encoding='utf8') as datafile:
    datafile.writelines(dataset_reduced)

