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
        self.trig=0


class WaveForm:
    def __init__(self, pmt, blw):
        self.blw=blw
        self.pmt=pmt

class Hit:
    def __init__(self, init, fin):
        self.init=init
        self.fin=fin
        self.groups=[]
        self.peaks=[] ## Reconstructed from the beginning of the group
        self.area=0
        self.height=0

class Group:
    def __init__(self, maxi, left, right, height):
        self.maxi=maxi
        self.left=left
        self.right=right
        self.height=height

class Peak:
    def __init__(self, pmt, maxi, h, blw):
        self.blw=blw
        self.init10=0
        self.maxi=maxi
        self.height=h
        self.init=0
        self.fin=0
        self.area=0
        self.pmt=pmt
