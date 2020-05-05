import numpy as np
Rec=[]
rec=np.recarray(3, dtype=[
    ('area', 'i8'),
    ])

for j in range(10):
    for i in range(3):
        rec[i]['area']=i*j
    Rec.extend(rec)
print(Rec[:]['area'])
