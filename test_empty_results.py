import pandas as pd
df = pd.read_csv('results/step4/20260509_074430/esc50/fewshot_results.csv')
df_err = df[df.metric == 'error']
print(df_err.groupby('value').size())
