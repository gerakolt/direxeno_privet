import numpy as np

def make_pmts():

    r=22/40

    pmt_mid=np.array([[1/np.sqrt(2)*np.cos(120*np.pi/180), 1/np.sqrt(2)*np.sin(120*np.pi/180), 1/np.sqrt(2)],
                    [-1/np.sqrt(2), 0, 1/np.sqrt(2)],
                    [1/np.sqrt(2)*np.cos(240*np.pi/180), 1/np.sqrt(2)*np.sin(240*np.pi/180), 1/np.sqrt(2)],
                    [0,1,0],
                    [-1/np.sqrt(2), 1/np.sqrt(2), 0],
                    [-1,0,0],
                    [-1/np.sqrt(2), -1/np.sqrt(2),0],
                    [1/np.sqrt(2)*np.cos(120*np.pi/180), 1/np.sqrt(2)*np.sin(120*np.pi/180), -1/np.sqrt(2)],
                    [-1/np.sqrt(2), 0, -1/np.sqrt(2)],
                    [1/np.sqrt(2)*np.cos(240*np.pi/180), 1/np.sqrt(2)*np.sin(240*np.pi/180), -1/np.sqrt(2)],
                    [1/np.sqrt(2)*np.cos(300*np.pi/180), 1/np.sqrt(2)*np.sin(300*np.pi/180), 1/np.sqrt(2)],
                    [1/np.sqrt(2), 0,1/np.sqrt(2)],
                    [1/np.sqrt(2)*np.cos(60*np.pi/180), 1/np.sqrt(2)*np.sin(60*np.pi/180), 1/np.sqrt(2)],
                    [0, -1, 0],
                    [1/np.sqrt(2), -1/np.sqrt(2), 0],
                    [1, 0, 0],
                    [1/np.sqrt(2), 1/np.sqrt(2), 0],
                    [1/np.sqrt(2)*np.cos(300*np.pi/180), 1/np.sqrt(2)*np.sin(300*np.pi/180), -1/np.sqrt(2)],
                    [1/np.sqrt(2), 0, -1/np.sqrt(2)],
                    [1/np.sqrt(2)*np.cos(60*np.pi/180), 1/np.sqrt(2)*np.sin(60*np.pi/180), -1/np.sqrt(2)],
                    ])


    pmt_r=np.zeros((len(pmt_mid),3))
    pmt_l=np.zeros((len(pmt_mid),3))
    pmt_up=np.zeros((len(pmt_mid),3))
    pmt_dn=np.zeros((len(pmt_mid),3))

    for i in range(len(pmt_mid)):
        a=pmt_mid[i][0]
        b=pmt_mid[i][1]
        pmt_r[i]=[-r*b/np.sqrt(a**2+b**2), r*a/np.sqrt(a**2+b**2), 0]
        pmt_l[i]=[r*b/np.sqrt(a**2+b**2),-r*a/np.sqrt(a**2+b**2), 0]
        pmt_up[i]=np.cross(pmt_mid[i], pmt_r[i])
        pmt_dn[i]=np.cross(pmt_mid[i], pmt_l[i])
    return pmt_mid, pmt_r, pmt_l, pmt_up, pmt_dn, r
