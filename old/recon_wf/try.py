import numpy as np

rec=np.recarray(3, dtype=[
    ('bool', '?'),
    ('ind', 'i8'),
    ('mat', 'f8', (2,2))])


for j in range(len(rec)):
    rec[j]=False, 1, np.ones((2,2))
    print(rec[j])
