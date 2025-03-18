# Drug_Response_Prediction

Welcome! This is the git repository of the paper "Promoting further research and progress in drug response prediction data and models".

To run the workflow, please do the following:

First, install [Snakemake](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html) if you haven't already. (The version we used is 7.8.5.) Then, you can set the configurations like the files to process, solvers to use, etc. in config.yaml.
After done configurating, you can run
`snakemake --use-conda --cores <number of cores or 'all' (without quotation marks)>` for cross-validation.
(If the command snakemake is not recognized, you may need to activate your snakemake environment with conda activate snakemake.)

For questions regarding the data analysis from the discussion section or the data preprocessing, see the notebook data/examine_data.ipynb or data/preprocess_data.ipynb, respectively.
How to download all data files is described in data/preprocess_data.ipynb. Some files can be found in this git repository, files that were too large and need to be downloaded manually are:
- data/CCLE_Data/OmicsSomaticMutations.csv
- data/CCLE_Data/OmicsCNGene.csv
- data/CCLE_Data/OmicsExpressionProteinCodingGenesTPMLogp1.csv
- data/Bulk_Cell_line_Genomic_Data/Cell_Model_Passports/MUT/mutations_all_20230202.csv
- data/Bulk_Cell_line_Genomic_Data/Cell_Model_Passports/CNV/cnv_20191101/cnv_gistic_20191101.csv
- data/Bulk_Cell_line_Genomic_Data/Cell_Model_Passports/GE/rnaseq_all_20220624/rnaseq_tpm_20220624.csv
- data/GDSC_Genetic_Features/PANCANCER_Genetic_features_GDSC2.csv
- DepMap_Data/TGSA_Data/protein.links.detailed.v11.0.txt (not described in data/preprocess_data.ipynb, but see line 70 in scripts/functions.py)

For questions regarding running the five repetitions mentioned in the end of the results section, see scripts/repetitions_evaluation.py.

For questions regarding obtaining the same figures as in the paper containing the results for each separate kind of data, (e.g., applying MMLP on gene expression data only and plotting it in the figure as MMLPexp), see for example scripts/MMLPexp.py.

The x-axis labels in the paper with the sub- and superscripts were modified using Inkscape.
