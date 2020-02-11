class DataSet:
    def __init__(self, label):
        self.events=[]
        self.label=label
        self.first_event=0
        self.last_event=0


class Event:
    def __init__(self,id):
        self.id=id
        self.wf=[]
        self.missing_PE=0
        self.PE=[]
        self.PE_std=[]
        self.pe_by_area=0
        self.time=0
        self.bins=[]
        self.parameters_fit=[]
        self.parameters_max=[]
        self.chi2=0
        self.Q=0
        self.T=0

class WaveForm:
    def __init__(self, pmt, blw):
        self.blw=blw
        self.hits=[]
        self.pmt=pmt
        self.init10=0


class Hit:
    def __init__(self, init, fin):
        self.init=init
        self.fin=fin
        self.groups=[]
        self.peaks=[] ## Reconstructed from the beginning of the group
        self.APs=[[],[],[]]
        self.area=0
        self.height=0

class Group:
    def __init__(self, maxi, left, right, height):
        self.maxi=maxi
        self.left=left
        self.right=right
        self.height=height

class Peak:
    def __init__(self, peak, amp, tau, sigma):
        self.peak=peak
        self.tau=tau
        self.sigma=sigma
        self.amp=amp
