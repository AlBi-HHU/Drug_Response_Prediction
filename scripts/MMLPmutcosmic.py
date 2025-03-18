"""
this file only exists for having the MMLPcosmicmut in the plots for the paper together with the other solvers
the MMLPcosmicmut results in the paper can be obtained by running MMLPcosmic on mut_all data only
(in config.yaml, set inputs: [mut_all.drug_smiles_GDSC2.pancancer_ic_pubchem_gdsc2]
instead of [cnv_all-exp_tpm-mut_all.drug_smiles_GDSC2.pancancer_ic_pubchem_gdsc2])

after running the workflow finishes, navigate to mut_all.drug_smiles_GDSC2.pancancer_ic_pubchem_gdsc2/recordwise (or subjectwise)
and rename the folder MMLPcosmic to MMLPcosmicmut and all files within from MMLPcosmic... to MMLPcosmicmut...
and move it into the folder cnv_all-exp_tpm-mut_all.drug_smiles_GDSC2.pancancer_ic_pubchem_gdsc2
(the rename.py script might help with this)

then run snakemake -s Snakefile_cv --use-conda --cores all --touch
so that snakemake does not try to overwrite the folder
"""