"""
this file only exists for having the MMLPcnv in the plots for the paper together with the other solvers
the MMLPcnv results in the paper can be obtained by running MMLP on cnv_tpm data only
(in config.yaml, set inputs: [cnv_tpm.drug_smiles_GDSC2.pancancer_ic_pubchem_gdsc2]
instead of [cnv_all-exp_tpm-mut_all.drug_smiles_GDSC2.pancancer_ic_pubchem_gdsc2])

after running the workflow finishes, navigate to cnv_tpm.drug_smiles_GDSC2.pancancer_ic_pubchem_gdsc2/recordwise (or subjectwise)
and rename the folder MMLP to MMLPcnv and all files within from MMLP... to MMLPcnv...
and move it into the folder cnv_all-exp_tpm-mut_all.drug_smiles_GDSC2.pancancer_ic_pubchem_gdsc2
(the rename.py script might help with this)

then run snakemake -s Snakefile_cv --use-conda --cores all --touch
so that snakemake does not try to overwrite the folder
"""