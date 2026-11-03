"""Export the consolidated competition score table from saved class metrics."""
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1]
    parts=[]
    for folder in ['baseline','transfer_alignment','transfer_improved','transfer_grid_search','improvement_25']:
        c=pd.read_csv(root/'data'/folder/'class_metrics.csv')
        c=c.rename(columns={'strategy':'method'})
        if 'city' not in c: c['city']='Amsterdam'
        if 'regime' not in c: c['regime']='original'
        if 'samples_per_class' not in c: c['samples_per_class']=25
        c['method']=c.method.replace({'source_target_blend':'original_blend'})
        keys=['city','regime','method','samples_per_class','fold','repetition']
        rows=[]
        for key,g in c.groupby(keys):
            assert set(g.age_class)=={1,2,3,4}
            rows.append({**dict(zip(keys,key)),'macro_f1':g.f1.mean(),
                         'weighted_f1':np.average(g.f1,weights=g.support),
                         **{f'class_{int(r.age_class)}_f1':r.f1 for r in g.itertuples()}})
        run=pd.DataFrame(rows)
        vals=['macro_f1','weighted_f1']+[f'class_{i}_f1' for i in range(1,5)]
        fold=run.groupby(keys[:-1],as_index=False)[vals].mean()
        scores=fold.groupby(keys[:-2],as_index=False)[vals].mean()
        sd=fold.groupby(keys[:-2]).macro_f1.std().rename('macro_f1_fold_sd').reset_index()
        scores=scores.merge(sd,on=keys[:-2]);scores['artifact_source']=folder
        parts.append(scores)
    all_scores=pd.concat(parts,ignore_index=True)
    k=['city','regime','method','samples_per_class']
    for _,g in all_scores.groupby(k):
        assert np.allclose(g.macro_f1,g.macro_f1.iloc[0]) and np.allclose(g.weighted_f1,g.weighted_f1.iloc[0])
    all_scores=all_scores.drop_duplicates(k).sort_values(k)
    all_scores.to_csv(root/'docs/results/all_model_scores.csv',index=False)
    print("Exported",len(all_scores),"model configurations to docs/results/all_model_scores.csv")


if __name__ == "__main__":
    main()
