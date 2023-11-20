<!---

    Copyright (c) 2023 Robert Bosch GmbH and its subsidiaries.

-->

# Experimental Resources for "BoschAI @ Causal News Corpus 2023: Robust Cause-Effect Span Extraction using Multi-Layer Sequence Tagging and Data Augmentation"

This repository contains the companion material for the following publication:

> Timo Pierre Schrader, Simon Razniewski, Lukas Lange, Annemarie Friedrich. **BoschAI @ Causal News Corpus 2023: Robust Cause-Effect Span Extraction using Multi-Layer Sequence Tagging and Data Augmentation.** CASE 2023.

Please cite this paper if using the dataset or the code, and direct any questions to
[Timo Schrader](mailto:timo.schrader@de.bosch.com).

## Purpose of this Software

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be maintained nor monitored in any way.

## Event Causality Identification with Causal News Corpus - Shared Task 3, CASE 2023
The "Event Causality Identification with Causal News Corpus" shared task, hosted by [Tan et al., (2023)](#1) at CASE 2023, aims at detecting and extracting causal relationships of form "X causes Y" within event-centric sentences. It builds upon the _Causal News Corpus_ (CNC, [Tan et al., (2022b)](#3)) and last year's iteration of the shared task [(Tan et al., 2022a)](#2). Our team, BoschAI, participated in this year's iteration and ranked 3<sup>rd</sup> in subtask 1 and 1<sup>st</sup> in subtask 2, outperforming the baseline by 13 percentage points in terms of F1 score.

### Subtask 1
Subtask 1 is a binary classification task that deals with the classification of sentences into _causal_ and _non-causal_. A sentence is labeled _causal_ if it contains _any_ cause-effect chain.

### Subtask 2
Subtask 2 is about extracting the exact token spans of _Cause_ (`<ARG0>`), _Effect_ (`<ARG1>`), and _Signal_ (`<SIG0>`). There are up to four different causal relations within a single sentence.

You can find more details in the cited papers as well as in the official [Github repository](https://github.com/tanfiona/CausalNewsCorpus).

## Our Approaches
To produce contextualized embeddings for the sentences in CNC, we use BERT-based models (namely BERT-Large [(Devlin et al., 2019)](#4) and RoBERTA-Large [(Liu et al., 2019)](#5)) in both subtasks. On top of that, we use augmented data to further train the models with additional samples. These synthetic samples are created using

* Easy Data Augmentation (EDA, [Wei et al., (2019)](#6))
* Oversampling
* ChatGPT

We find that RoBERTa-Large in combination with EDA augmented data works best and yields the best scores of all our experiments.

### Subtask 1
We use the logits produced by RoBERTa-Large to obtain probability estimates using a softmax function that acts on two output scores, one for each class. This allows us to use a weighted cross entropy loss during training.

### Subtask 2
On top of the BERT-based LM, we employ a conditional random field [(Lafferty et al., 2001)](#7) that produces consistent tagging sequences using the BILOU scheme. With that, we can extract up to three different causal relations and their exact positions in the text.

You can find all details of our modelling approaches in our paper.

## Results
We refer to the [public leaderboard](https://codalab.lisn.upsaclay.fr/competitions/11784).

In subtask 1, we rank 3<sup>rd</sup> with the following scores:
| # | User     | Precision     | Recall  | F1         |
|:-:|----------|------------|------------|------------|
| 1 | DeepBlueAI | 83.2     | 86.1 	  | 84.7    |
| 2 | InterosML | 81.6     | 87.3 	  | 84.4    |
| 3 | __BoschAI__ | 80.0     | 87.9	  | 83.3    |
| ... | ... | ...     | ... 	  | ...    |

In subtask 2, we rank 1<sup>st</sup> by a large margin:
| # | User     | Precision     | Recall  | F1         |
|:-:|----------|------------|------------|------------|
| 1 | __BoschAI__ | 84.4     | 64.0	  | 72.8   |
| 2 | tanfiona (baseline) | 60.3    | 59.2	  | 59.7   |
| 3 | CSECU-DSG | 40.0     | 36.1	  | 38.0    |


## Run the Code

To get things started, please create your Python environment from [environment.yml](environment.yml).

Next, download the original repository by [Tan et al., (2023)](#1) (https://github.com/tanfiona/CausalNewsCorpus) and place it in the root directory of this project as "CausalNewsCorpus" (make sure to not have a dangling "master" in the end of the name). We verified commit `455c3fb` to be functional with our code.

Afterward, you can run the two bash scripts provided in [scripts](scripts) (__run from within the [scripts](scripts) folder!__). All parameters are already set to reproduce our results from our paper. Feel free to modify them and test further configurations.
Please have a look at the code to find all available flags and their explanation.

__Note__: Due to license reasons, we do not publish the EDA augmented data that was used to get the leaderboard results. Refer to [augmented_data/README](augmented_data/README) to see how to generate the data yourself. When having it generated, you can set the path of the train files to these newly augmented files in the two bash scripts.

## License

This software is open-sourced under the MIT license. See the [LICENSE](LICENSE) file for details.

## Cite Us

If you use our software or dataset in your scientific work, please cite our paper:

```
@inproceedings{schrader-etal-2023-boschai,
    title = "{B}osch{AI} @ Causal News Corpus 2023: Robust Cause-Effect Span Extraction using Multi-Layer Sequence Tagging and Data Augmentation",
    author = "Schrader, Timo Pierre  and
      Razniewski, Simon  and
      Lange, Lukas  and
      Friedrich, Annemarie",
    editor = {H{\"u}rriyeto{\u{g}}lu, Ali  and
      Tanev, Hristo  and
      Zavarella, Vanni  and
      Yeniterzi, Reyyan  and
      Y{\"o}r{\"u}k, Erdem  and
      Slavcheva, Milena},
    booktitle = "Proceedings of the 6th Workshop on Challenges and Applications of Automated Extraction of Socio-political Events from Text",
    month = sep,
    year = "2023",
    address = "Varna, Bulgaria",
    publisher = "INCOMA Ltd., Shoumen, Bulgaria",
    url = "https://aclanthology.org/2023.case-1.5",
    pages = "38--43",
}
```

## Citations for the Causal News Corpus
<a id="1">[1]</a>
[Event Causality Identification with Causal News Corpus - Shared Task 3](https://github.com/tanfiona/CausalNewsCorpus) (Tan et al., CASE 2023)

<a id="2">[2]</a>
[Event Causality Identification with Causal News Corpus - Shared Task 3, CASE 2022](https://aclanthology.org/2022.case-1.28) (Tan et al., CASE 2022)

<a id="3">[3]</a>
[The Causal News Corpus: Annotating Causal Relations in Event Sentences from News](https://aclanthology.org/2022.lrec-1.246) (Tan et al., LREC 2022)

## Further Citations
<a id="4">[4]</a>
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423) (Devlin et al., NAACL 2019)

<a id="5">[5]</a>
[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) (Liu et al., 2019)

<a id="6">[6]</a>
[EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://aclanthology.org/D19-1670) (Wei & Zou, EMNLP-IJCNLP 2019)

<a id="7">[7]</a>
[Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](https://dl.acm.org/doi/10.5555/645530.655813) (Lafferty et al., ICML '01)
