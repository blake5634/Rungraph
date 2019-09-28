#!/usr/bin/python3
import pandas as pd
import pandas_profiling
x = pd.read_csv('ActivityLog.csv')
print('x is: ',type(x))
prof_rep = x.profile_report()
prof_rep.to_file(output_file='tmp.html')

