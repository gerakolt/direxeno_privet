import numpy as np

def make_pmts(PMTs):

    r=21/40

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


    pmt_r=np.zeros((len(PMTs),3))
    pmt_l=np.zeros((len(PMTs),3))
    pmt_up=np.zeros((len(PMTs),3))
    pmt_dn=np.zeros((len(PMTs),3))

    for i, pmt in enumerate(PMTs):
        a=pmt_mid[pmt][0]
        b=pmt_mid[pmt][1]
        pmt_r[i]=[-0.5*r*b/np.sqrt(a**2+b**2), 0.5*r*a/np.sqrt(a**2+b**2), 0]
        pmt_l[i]=[0.5*r*b/np.sqrt(a**2+b**2),-0.5*r*a/np.sqrt(a**2+b**2), 0]
        pmt_up[i]=np.cross(pmt_mid[pmt], pmt_r[i])
        pmt_dn[i]=np.cross(pmt_mid[pmt], pmt_l[i])
    return pmt_mid[PMTs], pmt_r, pmt_l, pmt_up, pmt_dn, r


def make_pmts_try():
    # r=21/40
    r=2
    mid=np.array([[1,0,0],[0,1,0], [-1,0,0], [0,-1,0], [0,0,1], [0,0,-1]])
    rt=np.ones_like(mid)
    up=np.ones_like(mid)
    rt[:-2]=np.cross(mid[:-2], np.array([0,0,1]))
    rt[-2:]=np.cross(mid[-2:], np.array([0,1,0]))
    # up[[0,2]]=np.cross(mid[[0,2]], np.array([0,1,0]))
    # up[[1,3,4,5]]=np.cross(mid[[1,3,4,5]], np.array([0,1,0]))
    up=np.cross(mid, rt)
    rt=(r*rt.T/np.sqrt(np.sum(rt**2, axis=1))/2).T
    up=(r*up.T/np.sqrt(np.sum(up**2, axis=1))/2).T

    return mid, rt, up
