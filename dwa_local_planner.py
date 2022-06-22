#!/usr/bin/python
# -*- coding: utf-8 -*-

################################################################################
#                                   Import                                     #
################################################################################
import math
import numpy as np


################################################################################
#                            Dynamic Window Approach                           #
################################################################################
class DWALocalPlanner():
    # ===========================================================================
    #   Constructor
    # ===========================================================================
    def __init__(self):
        print("\n================ DWA Local Planner =================")
        # robot parameter
        # rosparamで取得
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 1.0  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check
        

    # ===========================================================================
    #   Launch Server
    # ===========================================================================

    def launchServer(self):
        pass
        # pubsubの定義
        '''
        # TF Listener------------------------------------------------------------
        self.tf_listener = tf.TransformListener()

        # publisher
        # 速度指令 Publisher ----------------------------------------------------
        self.cmd_vel_publisher = rospy.Publisher(self.velocity_command_topic_name, Twist, queue_size=1)
        # target Publisher ------------------------------------------------------
        self.target_Publisher = rospy.Publisher(self.trace_target_topic_name, PoseStamped, queue_size=3)
        # 確認用 Publisher ------------------------------------------------------
        #self.area_publisher         = rospy.Publisher('fuzzy_area', std_msgs.msg.Float32MultiArray, queue_size=1)
        #self.mbsp_t_goal_publisher  = rospy.Publisher('mbsp_t_goal', std_msgs.msg.Float32MultiArray, queue_size=1)
        #self.mbsp_t_obst_publisher  = rospy.Publisher('mbsp_t_obst', std_msgs.msg.Float32MultiArray, queue_size=1)
        #self.mbsp_t_mix_publisher   = rospy.Publisher('mbsp_t_mix', std_msgs.msg.Float32MultiArray, queue_size=1)

        # Subscriber
        # Current pose Subscriber ----------------------------------------------
        command_subscriber = rospy.Subscriber(self.fuzzy_controller_command_topic_name, ControllerCommand, self.commandCallback)
        # LRS Data Subscriber --------------------------------------------------＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊要修正
        lrs_subscriber = rospy.Subscriber(self.laser_scan_topic_name, LaserScan, self.lrsCallback)

        # Timer
        rospy.Timer(rospy.Duration(0.05), self.controlFunc)
        # timerでループ

        print("Ready to serve command.")
        rospy.spin()
        '''

    def dwa_control(self, x, goal, ob):
        """
        Dynamic Window Approach control
        """
        # obはLRFのデータ
        dw = self.calc_dynamic_window(x)

        u, trajectory = self.calc_control_and_trajectory(x, dw, goal, ob)

        return u, trajectory


    def motion(self, x, u, dt):
        """
        motion model
        Whillは回転中心が差動二輪の軸の中心ではないことに注意
        """

        x[2] += u[1] * dt
        x[0] += u[0] * math.cos(x[2]) * dt
        x[1] += u[0] * math.sin(x[2]) * dt
        x[3] = u[0]
        x[4] = u[1]

        return x


    def calc_dynamic_window(self, x):
        """
        calculation dynamic window based on current state x
        """

        # Dynamic window from robot specification
        Vs = [self.min_speed, self.max_speed,
            -self.max_yaw_rate, self.max_yaw_rate]

        # Dynamic window from motion model
        Vd = [x[3] - self.max_accel * self.dt,
            x[3] + self.max_accel * self.dt,
            x[4] - self.max_delta_yaw_rate * self.dt,
            x[4] + self.max_delta_yaw_rate * self.dt]

        #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
            max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw


    def predict_trajectory(self, x_init, v, y):
        """
        predict trajectory with an input
        """

        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= self.predict_time:
            x = self.motion(x, [v, y], self.dt)
            trajectory = np.vstack((trajectory, x))
            time += self.dt

        return trajectory


    def calc_control_and_trajectory(self, x, dw, goal, ob):
        """
        calculation final input with dynamic window
        """

        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])

        # evaluate all trajectory with sampled input in dynamic window
        for v in np.arange(dw[0], dw[1], self.v_resolution):
            for y in np.arange(dw[2], dw[3], self.yaw_rate_resolution):

                trajectory = self.predict_trajectory(x_init, v, y)
                # calc cost
                to_goal_cost = self.to_goal_cost_gain * self.calc_to_goal_cost(trajectory, goal)
                speed_cost = self.speed_cost_gain * (self.max_speed - trajectory[-1, 3])
                ob_cost = self.obstacle_cost_gain * self.calc_obstacle_cost(trajectory, ob)

                final_cost = to_goal_cost + speed_cost + ob_cost

                # search minimum trajectory
                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, y]
                    best_trajectory = trajectory
                    if abs(best_u[0]) < self.robot_stuck_flag_cons \
                            and abs(x[3]) < self.robot_stuck_flag_cons:
                        # to ensure the robot do not get stuck in
                        # best v=0 m/s (in front of an obstacle) and
                        # best omega=0 rad/s (heading to the goal with
                        # angle difference of 0)
                        best_u[1] = -self.max_delta_yaw_rate
        return best_u, best_trajectory


    def calc_obstacle_cost(self, trajectory, ob):
        """
        calc obstacle cost inf: collision
        """
        ox = ob[:, 0]
        oy = ob[:, 1]
        dx = trajectory[:, 0] - ox[:, None]
        dy = trajectory[:, 1] - oy[:, None]
        r = np.hypot(dx, dy)

        # 障害物までの距離を計算

        min_r = np.min(r)
        return 1.0 / min_r  # OK


    def calc_to_goal_cost(self, trajectory, goal):
        """
            calc to goal cost with angle difference
        """

        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

        return cost

    # ===========================================================================
    #   Control Handler
    # ===========================================================================
    def controlFunc(self, event):
        print(__file__ + " start!!")
        # goal座標はサブスクライブする
        gx=10.0
        gy=10.0
        
        # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
        # goal position [x(m), y(m)]
        goal = np.array([gx, gy])

        # input [forward speed, yaw_rate]
        trajectory = np.array(x)
        ob = self.ob
        
        # process
        u, predicted_trajectory = self.dwa_control(x, goal, ob)
        x = self.motion(x, u, self.dt)  # simulate robot
        trajectory = np.vstack((trajectory, x))  # store state history

        # check reaching goal
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= self.robot_radius:
            print("Goal!!")
            # 終了フラグをパブリッシュ？終了判定はtracerでやるか？


################################################################################
#                               Main Function                                  #
################################################################################
if __name__ == '__main__':

    # Initialize node ------------------------------------------------------
    # rospy.init_node("dwa_local_planner")
    controller = DWALocalPlanner()
    controller.launchServer()

