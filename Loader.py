import re
import numpy as np

class IrisLoader:
    def __init__(self, filename, openmode = "rb"):
        self.filename = filename
        self.openmode = openmode

    def load(self):
        file = open(self.filename,self.openmode)
        result = {}
        file.readline()
        for line in file:
            sample = re.split(r",",line)
            sample[4] = sample[4].strip()
            if sample[4] in result:
                result[sample[4]].append(np.array(map(float,sample[0:4])))
            else:
                result[sample[4]] = [np.array(map(float,sample[0:4]))]

        file.close()

        return result
