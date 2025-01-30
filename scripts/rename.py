# for renaming files for the plots of MMLPmut/MMLPcnv/MMLPexp/MMLPmutcosmic/MMLPcnvcosmic/MMLPexpcosmic
import os

os.chdir("..")
os.chdir("5fold_cv/exp_tpm.drug_smiles_GDSC2.pancancer_ic_pubchem_gdsc2") # or cnv_all/mut_all instead of exp_tpm
x = "MMLP" # or MMLPcosmic
y = "MMLPexp" # or MMLPmut/MMLPcnv/MMLPmutcosmic/MMLPcnvcosmic/MMLPexpcosmic
for root, dir, files in os.walk("."):
    for d in dir:
        if x in d:
            d2 = d.replace(x, y)
            os.rename(os.path.join(root, d), os.path.join(root, d2))
for root, dir, files in os.walk("."):
    for f in files:
        if x in f:
            f2 = f.replace(x, y)
            os.rename(os.path.join(root, f), os.path.join(root, f2))