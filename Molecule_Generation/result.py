import re
import pandas as pd

results = []
with open('result_owndata.txt','r') as f:
    lst = f.readlines()
    for s in lst:
        if len(s[:-1]) >= 25:
            results.append(s[:-1])
    
results = pd.DataFrame(results)
results.to_csv('result.csv',index=False,header=False)