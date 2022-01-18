import time
import pybullet as pb
import pybullet_data

def main():
    client = pb.connect(pb.GUI) # or pb.DIRECT for headless

    pb.setGravity(0,0,-9.8)

    # add ground plane
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = pb.loadURDF("plane.urdf")

    # add objects
    cube_id = pb.loadURDF("cube.urdf", basePosition=(0,0,1), globalScaling=0.1)

    # keep script running to view GUI
    while 1:
        pb.stepSimulation()
        time.sleep(1./256)
        pass

if __name__ == "__main__":
    main()
