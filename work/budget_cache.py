from collections import OrderedDict
import threading
import numpy as np

def canonical(distances,ids):
    order=np.lexsort((ids,distances))
    return distances[order].copy(),ids[order].copy()

def prefix_admitted(ef,k,stored_k,distances):
    if k>stored_k or max(ef,k)!=max(ef,stored_k): return False
    if k==stored_k: return True
    return bool(distances[k-1]<distances[k])

class BudgetCache:
    def __init__(self,capacity):
        self.capacity=capacity
        self.entries=OrderedDict()
        self.lock=threading.Lock()
        self.hits=0
        self.misses=0
        self.evictions=0
        self.rejected=0
        self.prefix_hits=0

    def get(self,key,k,ef,guarded):
        with self.lock:
            value=self.entries.get(key)
            if value is None:
                self.misses+=1
                return None
            d,i=value
            admitted=prefix_admitted(ef,k,len(i),d) if guarded else k==len(i)
            if not admitted:
                self.misses+=1
                self.rejected+=1
                return None
            self.entries.move_to_end(key)
            self.hits+=1
            self.prefix_hits+=int(k<len(i))
            return d[:k].copy(),i[:k].copy()

    def put(self,key,value):
        with self.lock:
            old=self.entries.get(key)
            if old is None or len(old[1])<=len(value[1]): self.entries[key]=value
            self.entries.move_to_end(key)
            while len(self.entries)>self.capacity:
                self.entries.popitem(last=False)
                self.evictions+=1

    def payload_bytes(self):
        with self.lock:
            return sum(d.nbytes+i.nbytes for d,i in self.entries.values())
