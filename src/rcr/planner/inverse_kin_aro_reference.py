import pinocchio as pin 
import numpy as np
from numpy.linalg import pinv,inv,norm,svd,eig
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement

def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)

    q = qcurrent.copy()
    DT = 1e-1
    counter = 0
    while True:
        counter += 1
        pin.framesForwardKinematics(robot.model, robot.data, q)
        pin.computeJointJacobians(robot.model, robot.data, q)

        # get the current position of the hands and the cube
        oMlhand = robot.data.oMf[robot.model.getFrameId(LEFT_HAND)]
        oMrhand = robot.data.oMf[robot.model.getFrameId(RIGHT_HAND)]
        oMcubeL = getcubeplacement(cube, LEFT_HOOK)
        oMcubeR = getcubeplacement(cube, RIGHT_HOOK)

        # compute the error between the current position and the desired position
        lhandMcubeL = oMlhand.inverse() * oMcubeL
        rhandMcubeR = oMrhand.inverse() * oMcubeR

        # compute the velocity to reach the desired position
        lhand_nu = pin.log(lhandMcubeL).vector
        rhand_nu = pin.log(rhandMcubeR).vector
            
        # compute the jacobian of the hands
        lhand_J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.model.getFrameId(LEFT_HAND))
        rhand_J = pin.computeFrameJacobian(robot.model, robot.data, q, robot.model.getFrameId(RIGHT_HAND))


        lhand_vq = pinv(lhand_J) @ lhand_nu
        Plhand = np.eye(robot.model.nv) - pinv(lhand_J) @ lhand_J
        vq = lhand_vq + pinv(rhand_J @ Plhand) @ (rhand_nu - rhand_J @ lhand_vq)

        q = pin.integrate(robot.model, q, vq*DT)
        if viz is not None:
            viz.display(q)
        
        if counter % 5000 == 0:
            print("\niteration", counter)
            print("lhand_nu", norm(lhand_nu))
            print("rhand_nu", norm(rhand_nu))

        if norm(lhand_nu) < EPSILON and norm(rhand_nu) < EPSILON:
            break

    return q, not collision(robot, q) # and norm(lhand_nu) < EPSILON and norm(rhand_nu) < EPSILON
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()

    random_q = np.random.rand(robot.model.nq)

    viz.display(random_q)
    import time
    time.sleep(3)
    
    q = robot.q0.copy()
    q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)

    updatevisuals(viz, robot, cube, qe)