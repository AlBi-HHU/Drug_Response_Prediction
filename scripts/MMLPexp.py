"""
this file only exists for having the MMLPexp in the plots for the paper together with the other solvers
the MMLPexp results in the paper can be obtained by running MMLP on exp_tpm data only
(in config.yaml, set inputs: [exp_tpm.drug_smiles_GDSC2.pancancer_ic_pubchem_gdsc2]
instead of [cnv_all-exp_tpm-mut_all.drug_smiles_GDSC2.pancancer_ic_pubchem_gdsc2])

after running the workflow finishes, navigate to exp_tpm.drug_smiles_GDSC2.pancancer_ic_pubchem_gdsc2/recordwise (or subjectwise)
and rename the folder MMLP to MMLPexp and all files within from MMLP... to MMLPexp...
and move it into the folder cnv_all-exp_tpm-mut_all.drug_smiles_GDSC2.pancancer_ic_pubchem_gdsc2
(the rename.py script might help with this)

then run snakemake -s Snakefile_cv --use-conda --cores all --touch
so that snakemake does not try to overwrite the folder
"""