configfile:
    'config.yaml'

N = list(range(config['n_splits']))

rule all:
    input:
        expand(str(config['n_splits']) + 'fold_cv/{inputs}/resampling_{resampling}/{metric}plot.pdf', inputs=config['inputs'], resampling=config['resampling'], metric=config['metric'])

rule cv_split:
    output:
        expand(str(config['n_splits']) + 'fold_cv/{inputs}/resampling_{resampling}/splits/y_test_split{k}.npy', k=N, allow_missing=True),
        expand(str(config['n_splits']) + 'fold_cv/{inputs}/resampling_{resampling}/splits/y_val_split{k}.npy', k=N, allow_missing=True)
    params:
        cpus = '1',
        mem = '1G',
        gpus = '0',
        walltime = '00:00:59'
    conda:
        config['envfile']
    script:
        'scripts/cv_split.py'

rule model:
    input:
        str(config['n_splits']) + 'fold_cv/{inputs}/resampling_{resampling}/splits/y_val_split{k}.npy',
        str(config['n_splits']) + 'fold_cv/{inputs}/resampling_{resampling}/splits/y_test_split{k}.npy'
    output:
        str(config['n_splits']) + 'fold_cv/{inputs}/resampling_{resampling}/{s}/{s}_val_split{k}_RMSEs.txt',
        str(config['n_splits']) + 'fold_cv/{inputs}/resampling_{resampling}/{s}/{s}_test_split{k}_RMSEs.txt',
        str(config['n_splits']) + 'fold_cv/{inputs}/resampling_{resampling}/{s}/{s}_test_split{k}_CCCs.txt',
        str(config['n_splits']) + 'fold_cv/{inputs}/resampling_{resampling}/{s}/{s}_split{k}_best_model.pt'
    params:
        cpus = '1',
        mem = '1G',
        gpus = '0',
        walltime = '00:00:59'
    conda:
        config['envfile']
    script:
        'scripts/{wildcards.s}.py'

rule plot_results:
    input:
        expand(str(config['n_splits']) + 'fold_cv/{inputs}/resampling_{resampling}/{s}/{s}_test_split{k}_{metric}s.txt', s=config['solver'], k=N, allow_missing=True)
    output:
        str(config['n_splits']) + 'fold_cv/{inputs}/resampling_{resampling}/{metric}plot.pdf'
    params:
        cpus = '1',
        mem = '1G',
        gpus = '0',
        walltime = '00:00:59'
    conda:
        config['envfile']
    script:
        'scripts/plot_results.py'