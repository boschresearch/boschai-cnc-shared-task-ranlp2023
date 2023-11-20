# Data Augmentation

Due to license reasons, we cannot publish our augmented data, so please create them yourself using [EDA](https://github.com/jasonwei20/eda_nlp) and create the following files using the following parameters:

* __eda_train_subtask1.csv__: Based on [CausalNewsCorpus/data/V2/train_subtask1.csv](../CausalNewsCorpus/data/V2/train_subtask1.csv), apply $\alpha_{sr} = 0.4, \alpha_{ri} = 0.1, \alpha_{rs}=0.6$ with seed = 1 (default of EDA) and `--num_aug=5`. Make sure to end up with __18451__ samples in the CSV file.
* __eda_train_subtask2_grouped.csv__: Based on [CausalNewsCorpus/data/V2/train_subtask2_grouped.csv](../CausalNewsCorpus/data/V2/train_subtask2_grouped.csv), apply $\alpha_{sr} = 0.4, \alpha_{ri} = 0.5$ with seed = 1 (default of EDA) and `--num_aug=1`. __Only augment 1-relation instances__. Make sure to end up with __4612__ samples in the CSV files.
