import pybullet as p
import time
import pybullet_data
from task_environment import *

#p.connect(p.UDP,"192.168.86.100")

cid = p.connect(p.SHARED_MEMORY)
if (cid < 0):
  p.connect(p.GUI)
# p.setInternalSimFlags(0) 
p.resetSimulation()

# p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.loadURDF("plane.urdf")
p.setRealTimeSimulation(1)
CONTROLLER_ID = 0
POSITION = 1
ORIENTATION = 2
ANALOG = 3
BUTTONS = 6

# frameNr = 0
# objectInfo = ""
pointRay = -1
pointRayx = -1
rayLen = 1000

while True:
    events = p.getVREvents(p.VR_DEVICE_GENERIC_TRACKER)
    # events = p.getVREvents()
    print(events)
    for e in (events):
       print(e)
        