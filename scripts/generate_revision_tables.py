#!/usr/bin/env python3
"""Generate manuscript revision tables from the frozen and post-review records."""
from pathlib import Path
import csv
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / 'artifacts/paper/revision/analysis'
OUTPUT = ROOT / 'artifacts/paper/revision/Manuscript_Source/generated/tables'
NAMES = {'cora': 'Cora', 'citeseer': 'CiteSeer', 'pubmed': 'PubMed', 'flickr': 'Flickr'}

def table(path, columns, header, rows):
    path.write_text('\n'.join([r'\begin{tabular}{'+columns+'}', r'\toprule',
                  header + r'\\', r'\midrule', *[r' & '.join(row)+r'\\' for row in rows],
                  r'\bottomrule', r'\end{tabular}', '']))

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paired = list(csv.DictReader((ANALYSIS/'statistics_paired_summary.csv').open()))
    for comparison, filename in [('learned_vs_full','paired_main.tex'),('learned_vs_pca','paired_pca.tex')]:
        rows = []
        for row in paired:
            if row['comparison'] != comparison:
                continue
            values = [NAMES[row['dataset']]]
            if comparison == 'learned_vs_pca':
                values.append(row['embedding_dim'])
            values += [f"{float(row['mean_accuracy_difference']):+.4f}",
                       f"[{float(row['paired_t_ci95_lower']):+.4f}, {float(row['paired_t_ci95_upper']):+.4f}]",
                       f"{float(row['exact_two_sided_sign_flip_p']):.4f}",
                       f"{float(row['holm_p_eight_exploratory_comparisons']):.4f}"]
            rows.append(values)
        is_pca = comparison == 'learned_vs_pca'
        table(OUTPUT/filename, 'l'+('r' if is_pca else '')+'rrrr',
              'Dataset'+(' & $k$' if is_pca else '')+r' & Difference & 95\% paired CI & Exact $p$ & Holm $p$', rows)
    placement = json.loads((ANALYSIS/'placement_results.json').read_text())
    lookup = {(row['dataset'],row['num_partitions'],row['strategy']):row for row in placement['summary']}
    rows = []
    for dataset in NAMES:
        for count in (2,4,8):
            values = [NAMES[dataset],str(count)]
            for metric, scale in [('boundary_edge_ratio',1),('serialized_wire_bytes',1e6)]:
                for strategy in ('random','rcm','metis'):
                    row=lookup[dataset,count,strategy]
                    value=f"{row[metric+'_mean']/scale:.3f}"
                    if strategy!='rcm':
                        value += r' $\pm$ '+f"{row[metric+'_sd']/scale:.3f}"
                    values.append(value)
            rows.append(values)
    header = r'Dataset & $P$ & Random cut & RCM cut & METIS cut & Random MB & RCM MB & METIS MB'
    table(OUTPUT/'placement_by_dataset.tex','lr'+'r'*6,header,rows)
    records=[json.loads(line) for line in (ROOT/'artifacts/paper/results/release-v1.0.0-final/raw/results.jsonl').open()]
    depth=[]
    depth_rows=[]
    for dataset in NAMES:
        selected=[row for row in records if row['dataset']==dataset and row['system']=='learned'
                  and row['embedding_dim']==64 and row['num_partitions']==4
                  and row['partition_strategy']=='random' and row['network_profile']=='constrained']
        traces={row['partition_seed']:row['serialized_wire_bytes'] for row in selected}
        assert len(traces)==3 and len(selected)==15
        central={row['serialized_wire_bytes'] for row in records if row['dataset']==dataset and row['system']=='full'}
        assert len(central)==1
        row={'dataset':dataset,'one_round_bytes':statistics.mean(traces.values()),'central_float32_bytes':central.pop()}
        values=[NAMES[dataset]]
        for layers in (1,2,4,8):
            ratio=row['central_float32_bytes']/(layers*row['one_round_bytes'])
            row[f'ratio_L{layers}']=ratio
            values.append(f'{ratio:.2f}')
        depth.append(row)
        depth_rows.append(values)
    table(OUTPUT/'depth.tex','lrrrr',r'Dataset & $L=1$ & $L=2$ & $L=4$ & $L=8$',depth_rows)
    with (ANALYSIS/'depth_extrapolation.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(depth[0]))
        writer.writeheader(); writer.writerows(depth)
    print('Generated 4 revision tables and depth analysis')

if __name__=='__main__':
    main()
