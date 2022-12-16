#!/usr/bin/env python

import argparse

import rospy
import os
import numpy as np
import intera_interface
import intera_external_devices
# from pytictoc import TicToc
from std_msgs.msg import String
from collections import deque

import osqp
from scipy import sparse


# timer = TicToc()

# import pyomo.environ as pyo
# from pyomo.opt import SolverStatus, TerminationCondition

from intera_interface import CHECK_VERSION

prev_point = np.zeros(2)
prev_angle_vel = np.zeros(2)

t = np.zeros(1)

alpha = 0.5

'''
input_angles = np.zeros((2, 501))

with open('pitch_roll_inputs_vel.txt') as f:
    i = 0
    for line in f:
        tokens = line.split()
        input_angles[0, i] = tokens[0]
        input_angles[1, i] = tokens[1]
        i += 1
'''
def solveMPC_osqp(x0, u0, xr):
  
    dt = 1/30
    g = 9.81
    angle_rate_limit = 0.15  
    angle_acc_limit = 0.01
    angle_limit = np.pi/6

    # Discrete time model of ball on plate
    Ad = sparse.csc_matrix([
    [1, 0, dt, 0, 0, 0],
    [0, 1, 0, dt, 0, 0],
    [0, 0, 1, 0, -dt*g, 0],
    [0, 0, 0, 1, 0, -dt*g],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    ])
    # print(Ad)
    Bd = sparse.csc_matrix([
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [dt, 0],
    [0, dt],
    ])
    # print(Bd)
    [nx, nu] = Bd.shape
    # print([nx, nu])

    # Constraints
    umin = np.array([-angle_rate_limit, -angle_rate_limit])
    umax = np.array([angle_rate_limit, angle_rate_limit])
    xmin = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -angle_limit, -angle_limit])
    xmax = np.array([np.inf, np.inf, np.inf, np.inf, angle_limit, angle_limit])

    # Objective function
    Q = sparse.diags([40., 40., 2., 2., 20., 20.]) # TUNE
    QN = Q
    R = 10*sparse.eye(2) # TUNE

    # Initial and reference states
    # xr = np.array([0.,0.,0.,0.,0.,0.]) # TUNE
    xr = xr

    # Prediction horizon
    N = 30

    # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
    # - quadratic objective
    P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                        sparse.kron(sparse.eye(N), R)], format='csc')
    # print(P)
    # print(P.shape)
    # print(Q.shape)
    # - linear objective
    q = np.hstack([np.kron(np.ones(N), -Q.dot(xr)), -QN.dot(xr),
                np.zeros(N*nu)])
    # print(q)
    # print(q.shape)
    # print(Q.dot(xr).shape)
    # - linear dynamics
    Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
    Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
    Aeq = sparse.hstack([Ax, Bu])
    leq = np.hstack([-x0, np.zeros(N*nx)])
    ueq = leq
    # - input and state constraints
    Aineq = sparse.eye((N+1)*nx + N*nu)
    lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
    uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])

    diagonals = [np.hstack([np.zeros((N+1)*nx), np.ones(N*nu)]), np.hstack([np.zeros((N+1)*nx), np.ones((N-1)*nu)*-1])]
    Aacc = sparse.diags(diagonals, [0, -nu])
    uacc = np.hstack([np.zeros((N+1)*nx), angle_acc_limit + u0[0], angle_acc_limit + u0[1], np.kron(np.ones((N-1)*nu), angle_acc_limit)])
    lacc = np.hstack([np.zeros((N+1)*nx), -angle_acc_limit + u0[0], -angle_acc_limit + u0[1], np.kron(np.ones((N-1)*nu), -angle_acc_limit)])

    # print(Aeq.shape)
    # print(Aineq.shape)
    # print(Aacc.shape)

    # - OSQP constraints
    A = sparse.vstack([Aeq, Aineq, Aacc], format='csc')
    l = np.hstack([leq, lineq, lacc])
    u = np.hstack([ueq, uineq, uacc])

    # print(u.shape)

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace
    prob.setup(P, q, A, l, u, warm_start=True, verbose=False)

    res = prob.solve()

    ctrl = res.x[-N*nu:-(N-1)*nu]
    # x0 = Ad.dot(x0) + Bd.dot(ctrl)
    x = res.x[0:(N+1)*nx:nx]
    y = res.x[1:(N+1)*nx:nx]

    return np.asarray(ctrl)

def reference(n):

    # rate = 0.05
    # amp = 0.07
    # shape = 1

    # x = amp*np.sin(rate*n)
    # y = amp*np.cos(shape*rate*n)
    # x_vel = amp*rate*np.cos(rate*n)
    # y_vel = -amp*shape*rate*np.sin(shape*rate*n)
    # pitch = 0.0
    # roll = 0.0

    # if n < 215:
    #     x = 0.145*np.cos(np.pi*n/215)
    #     y = 0.02 - 0.02*np.cos(2*np.pi*n/215)
    #     x_vel = -0.145*np.pi*np.sin(np.pi*n/215)/215
    #     y_vel = 0.02*2*np.pi*np.sin(2*np.pi*n/215)/215
    #     pitch = 0
    #     roll = 0
    # else:
    #     x = -14.5/100
    #     y = 0
    #     x_vel, y_vel = 0, 0
    #     pitch = 0
    #     roll = 0

    x = 0.11*np.cos(np.pi*n/215)
    y = 0.06*np.sin(np.pi*n/215)
    x_vel = -0.11*np.pi*np.sin(np.pi*n/215)/215
    y_vel = 0.06*np.pi*np.cos(np.pi*n/215)/215
    pitch = 0
    roll = 0

    return np.array([y, x, y_vel, x_vel, pitch, roll])

def joint_vel(data, args):

    limb = args[0]
    desp = args[1]
    desr = args[2]

    joint_care = ['right_j3', 'right_j4']

    current_position = np.array([limb.joint_angles()[joint_name] for joint_name in joint_care])
    current_position[0] = -(current_position[0] - desp)
    current_position[1] = current_position[1] - desr

    current_velocity = np.array([limb.joint_velocities()[joint_name] for joint_name in joint_care])
    current_velocity[0] = -current_velocity[0]
    print('pitch velocity:', current_velocity[0]) 
    print('roll  velocity:', current_velocity[1]) 

    point = data.data.split()

    y = -float(point[0])/100
    x = -float(point[1])/100

    x = alpha*x + prev_point[0]*(1-alpha)
    y = alpha*y + prev_point[1]*(1-alpha)

    velocity_x = (x - prev_point[0])*30
    velocity_y = (y - prev_point[1])*30

    print('Current Position : {} {}'.format(x, y))
    print('Current Velocity : {} {}'.format(round(velocity_x, 3), round(velocity_y, 3)))

    # print(round(velocity_y, 3))

    prev_point[0] = x
    prev_point[1] = y

    # print(prev_point)

    xr = reference(t[0])

    # timer.tic()
    pvel, rvel = solveMPC_osqp(np.array([x, y, velocity_x, velocity_y, current_position[0], current_position[1]]), prev_angle_vel, np.zeros(6))
    # timer.toc('Solve Time Took: ')

    prev_angle_vel[0] = pvel
    prev_angle_vel[1] = rvel

    dic_vel = {'right_j3':-pvel, 'right_j4':rvel}
    limb.set_joint_velocities(dic_vel)

    # print(dic_vel)

    print(pvel, rvel)

    t[0] = t[0] + 1
    print(t)

def main():
    """RSDK Joint Position Example: Keyboard Control

    Use your dev machine's keyboard to control joint positions.

    Each key corresponds to increasing or decreasing the angle
    of a joint on Sawyer's arm. The increasing and descreasing
    are represented by number key and letter key next to the number.
    """
    epilog = """
             See help inside the example with the '?' key for key bindings.
             """
    rp = intera_interface.RobotParams()
    valid_limbs = rp.get_limb_names()
    if not valid_limbs:
        rp.log_message(("Cannot detect any limb parameters on this robot. "
                        "Exiting."), "ERROR")
        return
    arg_fmt = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt,
                                     description=main.__doc__,
                                     epilog=epilog)
    parser.add_argument(
        "-l", "--limb", dest="limb", default=valid_limbs[0],
        choices=valid_limbs,
        help="Limb on which to run the joint position keyboard example"
    )
    args = parser.parse_args(rospy.myargv()[1:])

    print("Initializing node... ")
    rospy.init_node("sdk_joint_position_keyboard")
    print("Getting robot state... ")
    rs = intera_interface.RobotEnable(CHECK_VERSION)
    init_state = rs.state().enabled

    def clean_shutdown():
        print("\nExiting example.")

    rospy.on_shutdown(clean_shutdown)

    rospy.loginfo("Enabling robot...")
    rs.enable()

    limb = intera_interface.Limb(args.limb)

    desp = limb.joint_angles()['right_j3']
    desr = limb.joint_angles()['right_j4']

    # joint_vel(args.limb)
    rospy.Subscriber("string_pub", String, joint_vel, (limb, desp, desr))
    # joint_vel()
    print("Done.")
    rospy.spin()


if __name__ == '__main__':
    main()
    f.close()
