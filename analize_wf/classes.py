class WaveForm:
    def __init__(self, pmt, blw):
        self.blw=blw
        self.hits=[]
        self.pmt=pmt
        self.init10=0


class Hit:
    def __init__(self, init, fin):
        self.init=init
        self.init10=0
        self.fin=fin
        self.groups=[]
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
