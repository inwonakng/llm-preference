import pandas as pd
from pathlib import Path

print('#' * 30)
print('3 way')
print('#' * 30)
summary = pd.concat([
    pd.read_csv(
        report,
        index_col='Unnamed: 0'
    ).assign(**{
        'Dataset': ' '.join([
            w.capitalize() 
            for w in report.parent.parent.parent.name.split('_')
        ]),
        'Setup': 'few_shot' if report.parent.name == 'with_example' else 'zero_shot'
    })
    for report in Path('evaluation').glob('*/*/*/3_way.csv')
    if report.parent.parent.parent.name != 'compsent'
])[['Dataset','Setup','F1 Micro','F1 Macro','F1 Weighted','F1[0]','F1[1]','F1[2]']].round(4)
print(summary)
summary.to_csv('3_way_summary.csv')

print('#' * 30)
print('4 way')
print('#' * 30)
summary = pd.concat([
    pd.read_csv(
        report,
        index_col='Unnamed: 0'
    ).assign(**{
        'Dataset': ' '.join([
            w.capitalize() 
            for w in report.parent.parent.parent.name.split('_')
        ]),
        'Setup': 'few_shot' if report.parent.name == 'with_example' else 'zero_shot'
    })
    for report in Path('evaluation').glob('*/*/*/4_way.csv')
    if report.parent.parent.parent.name != 'compsent'
])[['Dataset','Setup','F1 Micro','F1 Macro','F1 Weighted','F1[0]','F1[1]','F1[2]','F1[3]']].round(4)
print(summary)
summary.to_csv('4_way_summary.csv')
