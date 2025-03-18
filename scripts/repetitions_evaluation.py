# for getting 5 repetitions of the cross-validation, we changed the seed in line 18 in cv_split.py np.random.seed(0) from 0 to 10, 20, 30, 40, 50 and renamed the resulting directory cnv_all-exp_tpm-mut_all.drug_smiles_GDSC2.pancancer_ic_pubchem_gdsc2 to cnv_all-exp_tpm-mut_all.drug_smiles_GDSC2.pancancer_ic_pubchem_gdsc2_seed10 and so on
import os
import numpy as np

os.chdir('..')
os.chdir('5fold_cv')
RMSEs = []
CCCs = []
for solver in ['TGDRP', 'TGSA', 'mean_predictor']:
    RMSEs.append([])
    CCCs.append([])
    for seed in [10, 20, 30, 40, 50]:
        os.chdir(f'cnv_all-exp_tpm-mut_all.drug_smiles_GDSC2.pancancer_ic_pubchem_gdsc2_seed{seed}/resampling_targetwise/{solver}')
        for file in os.listdir():
            if 'test' in file:
                with open(file, 'r') as f:
                    if 'RMSE' in file:
                        RMSEs[-1].append(float(f.readline()))
                    if 'CCC' in file:
                        CCCs[-1].append(float(f.readline()))
        os.chdir('../../..')
RMSEs = np.array(RMSEs)
CCCs = np.array(CCCs)
print(np.mean(RMSEs, axis=1), np.std(RMSEs, axis=1))
print(np.mean(CCCs, axis=1), np.std(CCCs, axis=1))