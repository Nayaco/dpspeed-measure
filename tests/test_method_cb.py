class A:
    def __init__(self):
        self.count = 1
    def addone(self):
        self.count += 1
    def getcb(self):
        return self.addone
    
a = A()
cb = a.getcb()
cb()
print(a.count)