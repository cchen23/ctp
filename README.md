# ctp
Code for [_Constructing Taxonomies from Pretrained Language Models_](https://aclanthology.org/2021.naacl-main.373.pdf), presented at NAACL 2021.

## Installation

```
git clone git@github.com:cchen23/ctp.git
cd ctp
conda create -n ctp python=3.6
conda activate ctp
pip install -r requirements.txt
```

## Getting Started
To download the data. Download the [data](https://drive.google.com/file/d/1r0koth-KO1HVyBvphCwbDSrvYDil4KD_/view?usp=sharing). Place the `generated_training_pairs` folder in `wordnet_reconstruction/datasets/`, and place the `df_csvs` folder in `wordnet_reconstruction/datasets/data_creators/df_csvs`.

The CTP approach consists of two steps: parenthood prediction (described in Section 2.2 of our paper) and tree reconciliation (described in Section 2.3 of our paper).

To perform the parenthood prediction step for a model with the configuration file `experiment_name`, run:
```
cd scripts/
python run_finetuning_hypernym_classification_multidomain.py \
  --experiment-name [experiment_name]
```

To perform the tree reconciliation step, run:

```
cd ctp/inference/
python examine_subtrees.py --experiment-name [experiment_name] \
  --prediction-metric-type ancestor
```

## Recreating the Data.
### English dataset (From Bansal et al 2014):
First, create a CSV file with the Bansal et al 2014 dataset:
```
cd datasets/data_creators
python preprocess.py
```

Then, retrieve the web contexts and web definitions.
```
cd data_creators/
python get_contexts.py --cached-contexts-filename contexts_bansal_1004_with_merriam_webster.json --new-contexts-filename contexts_bansal_1004_with_merriam_webster.json --wordnet-trees-file ../datasets/data_creators/df_csvs/bansal14_trees.csv
```

Then, create the training examples.
```
cd datasets/data_creators
python create_wordnet_data.py --data-dir ./ --save-dir \ 
  ../texeval/generated_training_pairs --anc-label 0 --sib-label 0 \
   --desc-label 0 --rand-label 0 --parent-label 1 --retrieve
```

### Non-English datasets:
First, create a `.csv` file with the synset corresponding to each term in the Bansal et al 2014 dataset.
```
python get_bansal_tree_synset_names.py
```
For a few instances, the correct synset cannot be determined, so this .csv file needs to be manually edited.
The edited version is available in `df_csvs/bansal14_trees_synset_names_cleaned.csv` in the uploaded data [data](https://drive.google.com/file/d/1r0koth-KO1HVyBvphCwbDSrvYDil4KD_/view?usp=sharing).

Then, create the non-English training examples:
```
cd datasets/data_creators
python create_wordnet_data_mling_cleaned.py --data-dir ./df_csvs \
  --save-dir ../texeval/generated_training_pairs --anc-label 0 \
  --sib-label 0 --desc-label 0 --rand-label 0 --parent-label 1
```

## References
```
@inproceedings{chen-lin-klein:2021:NAACL,
  author={Chen, Catherine and Lin, Kevin and Klein, Dan},
  title={Constructing Taxonomies from Pretrained Language Models},
  booktitle={Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year={2021}
}
```
