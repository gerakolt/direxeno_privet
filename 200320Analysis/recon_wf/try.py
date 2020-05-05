import numpy as np

rec=np.recarray(2, dtype=[
    ('a', 'i8'),
    ('b', 'i8', 2),
    ])

for i in range(2):
    rec[i]['a']=i
    rec[i]['b']=[10*i+1, 10*i+2]

print(rec)
