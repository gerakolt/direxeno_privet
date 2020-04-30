import numpy as np

rec=np.recarray(0, dtype=[
    ('id', 'i8'),
    ('pmt', ('i8',3))])

for i in range(5):
    rec=np.append(rec, [(0, [1,2,3])])
print(rec)
