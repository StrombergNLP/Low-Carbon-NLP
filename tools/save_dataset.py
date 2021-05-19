from datasets import load_dataset

dataset = load_dataset('cc_news', script_version='master')

dataset_reduced = dataset['train']['text'][:100000]
print(len(dataset_reduced))
del dataset

with open('cc_news_reduced.txt', 'w+', encoding='utf8') as datafile:
    datafile.writelines(dataset_reduced)

