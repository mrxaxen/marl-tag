import rclpy
from rclpy.node import Node
import torch.nn as nn
from std_msgs.msg import Int64
from tutorial_interfaces.msg import Num  
from geometry_msgs.msg import Twist
import math
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import torch as T
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1, fc2,
                 n_actions, name, chkpt_dir):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.pi = nn.Linear(fc2, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = T.sigmoid(self.pi(x))

        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


def transform_orientation(orientation):
    # orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
    # (roll, pitch, yaw) = euler_from_quarternion(orientation_list)
    qx, qy, qz, qw = orientation.x, orientation.y, orientation.z, orientation.w
    roll = math.atan2(2*(qw*qx + qy*qz), 1-2*(qx**2 + qy**2))
    pitch = -math.pi/2 + 2*math.atan2(math.sqrt(1+2*(qw*qy-qx*qz)), math.sqrt(1-2*(qw*qy-qx*qz)))
    yaw = math.atan2(2*(qw*qz + qx*qy), 1-2*(qy**2 + qz**2))
    return roll, pitch, yaw


def distance(xy_1, xy_2):
    x_diff = math.pow((xy_1[0]-xy_2[0]), 2)
    y_diff = math.pow((xy_1[1]-xy_2[1]), 2)
    dist = math.sqrt(x_diff + y_diff)
    return dist


def calculate_relative_ditances(adv_0_xy, adv_1_xy, adv_2_xy, ag_0_xy):
        # other_pos.append(other.state.p_pos - agent.state.p_pos)
        # thus the relative distances are calculated from other pos - our pos
        # agent 0 <-> adv 0:  
        d_ag0_ad0_x = adv_0_xy[0] - ag_0_xy[0]
        d_ag0_ad0_y = adv_0_xy[1] - ag_0_xy[1]
        # agent 0 <-> adv 1:
        d_ag0_ad1_x = adv_1_xy[0] - ag_0_xy[0]
        d_ag0_ad1_y = adv_1_xy[1] - ag_0_xy[1]
        # agent 0 <-> adv 2:
        d_ag0_ad2_x = adv_2_xy[0] - ag_0_xy[0]
        d_ag0_ad2_y = adv_2_xy[1] - ag_0_xy[1]
        # adv 0 <-> adv 1:
        d_ad0_ad1_x = adv_1_xy[0] - adv_0_xy[0]
        d_ad0_ad1_y = adv_1_xy[1] - adv_0_xy[1]
        # adv 0 <-> adv 2:
        d_ad0_ad2_x = adv_2_xy[0] - adv_0_xy[0]
        d_ad0_ad2_y = adv_2_xy[1] - adv_0_xy[1]
        # adv 1 <-> adv 2:
        d_ad1_ad2_x = adv_2_xy[0] - adv_1_xy[0]
        d_ad1_ad2_y = adv_2_xy[1] - adv_1_xy[1]

        return (d_ag0_ad0_x, d_ag0_ad0_y, d_ag0_ad1_x, d_ag0_ad1_y, d_ag0_ad2_x, d_ag0_ad2_y, d_ad0_ad1_x, d_ad0_ad1_y, d_ad0_ad2_x, d_ad0_ad2_y, d_ad1_ad2_x, d_ad1_ad2_y) # 12 total


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.odom1 = []
        self.odom2 = []
        self.odom3 = []
        self.odom4 = []
        self.yaw1 = []
        self.yaw2 = []
        self.yaw3 = []
        self.yaw4 = []
        self.vel1 = []
        self.vel2 = []
        self.vel3 = []
        self.vel4 = []
        self.closest_quadrant1 = [0, 0] # which is closest quadrant, and what is the distance
        self.closest_quadrant2 = [0, 0]
        self.closest_quadrant3 = [0, 0]
        self.closest_quadrant4 = [0, 0]
        

        # scaling hyperparams to adjust
        self.obs_scale_factor = 4.5
        self.linear_vel_scale_factor = 0.04
        self.do_quad_control = False
        self.avoid_obstacles = True
        self.collision_boundary = 0.45
        self.stop_boundary = 0.15
        self.quandrants = False # switch between the two quadrant modes for laser stan distance
        

        # NN stuff
        self.device = 'cuda:0' if T.cuda.is_available() else 'cpu'
        self.agent_0_network = ActorNetwork(alpha=1e-4, input_dims=10, fc1=64, fc2=64, n_actions=5, # alpha is lr, shouldnt matter
                            name='agent_3_actor', chkpt_dir="/home/aaron/g_a/coll_int/src/robot_spawner_pkg/checkpoints/") # my path, change accordingly
    
        self.adversary_0_network = ActorNetwork(alpha=1e-4, input_dims=12, fc1=64, fc2=64, n_actions=5, # alpha is lr, shouldnt matter
                                name='agent_0_actor', chkpt_dir="/home/aaron/g_a/coll_int/src/robot_spawner_pkg/checkpoints/")
        ### trying loading adv_0 network for all adversaries, as it seemed to perform best
        self.adversary_1_network = self.adversary_0_network
        self.adversary_2_network = self.adversary_0_network
        """
        self.adversary_1_network = ActorNetwork(alpha=1e-4, input_dims=12, fc1=64, fc2=64, n_actions=5, # alpha is lr, shouldnt matter
                                name='agent_1_actor', chkpt_dir="/home/aaron/g_a/coll_int/src/robot_spawner_pkg/checkpoints/")
        
        self.adversary_2_network = ActorNetwork(alpha=1e-4, input_dims=12, fc1=64, fc2=64, n_actions=5, # alpha is lr, shouldnt matter
                                name='agent_2_actor', chkpt_dir="/home/aaron/g_a/coll_int/src/robot_spawner_pkg/checkpoints/")
        """
        self.agent_0_network.load_checkpoint()
        self.adversary_0_network.load_checkpoint()
        self.adversary_1_network.load_checkpoint()
        self.adversary_2_network.load_checkpoint()

        self.i = 0

        """
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.subscription  # prevent unused variable warning
        """
        self.subscription = self.create_subscription(
            Odometry,
            '/agent0/odom',            ### for now just odom of basic turtlebot
            self.odom_agent0_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.subscription = self.create_subscription(
            Odometry,
            '/adv1/odom',
            self.odom_adv1_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.subscription = self.create_subscription(
            Odometry,
            '/adv2/odom',
            self.odom_adv2_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.subscription = self.create_subscription(
            Odometry,
            '/adv3/odom',
            self.odom_adv3_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.subscription = self.create_subscription(
            LaserScan,
            '/agent0/scan',
            self.scan_agent0_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.subscription = self.create_subscription(
            LaserScan,
            '/adv1/scan',
            self.scan_adv1_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.subscription = self.create_subscription(
            LaserScan,
            '/adv2/scan',
            self.scan_adv2_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.subscription = self.create_subscription(
            LaserScan,
            '/adv3/scan',
            self.scan_adv3_callback,
            10)
        self.subscription  # prevent unused variable warning

        # self.dummy_pos_pub = self.create_publisher(String, "/dummy_pos", 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.agent_cmd_vel_pub = self.create_publisher(Twist, "/agent0/cmd_vel", 10)
        self.adv1_cmd_vel_pub = self.create_publisher(Twist, "/adv1/cmd_vel", 10)
        self.adv2_cmd_vel_pub = self.create_publisher(Twist, "/adv2/cmd_vel", 10)
        self.adv3_cmd_vel_pub = self.create_publisher(Twist, "/adv3/cmd_vel", 10)
        # self.timer_ = self.create_timer(0.5, self.send_velocity_command) # send command every 0.5s
        self.get_logger().info("Listening to odom node has been started")


    def call_network(self, obs):
        # reduce obs by a factor of 10, to conform to pettingzoo dims
        # adv0_obs = obs["adversary_0"]*0.1
        ag_obs = [self.obs_scale_factor * elem for elem in obs["agent_0"]]
        adv0_obs = [self.obs_scale_factor * elem for elem in obs["adversary_0"]]
        adv1_obs = [self.obs_scale_factor * elem for elem in obs["adversary_1"]]
        adv2_obs = [self.obs_scale_factor * elem for elem in obs["adversary_2"]]

        actions = {
                "adversary_0": self.adversary_0_network(T.tensor(adv0_obs, device=self.device)).detach().to("cpu").numpy(), # transform them into numpy arrays
                "adversary_1": self.adversary_1_network(T.tensor(adv1_obs, device=self.device)).detach().to("cpu").numpy(),
                "adversary_2": self.adversary_2_network(T.tensor(adv2_obs, device=self.device)).detach().to("cpu").numpy(),
                "agent_0": self.agent_0_network(T.tensor(ag_obs, device=self.device), ).detach().to("cpu").numpy(),
                }
        return actions

    # not currently used
    def dummy_odom_callback(self, msg):
        
        # message = msg
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        (posx, posy, posz) = (position.x, position.y, position.z)
        (qx, qy, qz, qw) = (orientation.x, orientation.y, orientation.z, orientation.w)
        
        # the pettingzoo movements are absolute directions, while cmd_vel is relative
        # therefore we need a transform based on our current dir 

        # in gazebo, "left" is positive Y
        # "up" is positive X
        # positive rotation about Z makes the robot turn left

        roll, pitch, yaw = transform_orientation(orientation=orientation)
        # relevant one is yaw

        message = Twist()
        message.linear.x = 0.
        message.linear.y = 0.
        message.linear.z = 0.
        message.angular.x = 0.
        message.angular.y = 0.
        message.angular.z = 0.
        # self.agent_cmd_vel_pub.publish(message)  
        # robot keeps the most recent mssg it recieved


    def odom_agent0_callback(self, msg):
        
        # message = msg
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        (posx, posy, posz) = (position.x, position.y, position.z)
        (qx, qy, qz, qw) = (orientation.x, orientation.y, orientation.z, orientation.w)
        lin_vel = msg.twist.twist.linear
        # ang_vel = msg.twist.twist.angular
        self.vel4.append([lin_vel.x, lin_vel.y])
        self.odom4.append([posx, posy])
        roll, pitch, yaw = transform_orientation(orientation=orientation)
        self.yaw4.append(yaw)


    def odom_adv1_callback(self, msg):
            
        # message = msg
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        (posx, posy, posz) = (position.x, position.y, position.z)
        (qx, qy, qz, qw) = (orientation.x, orientation.y, orientation.z, orientation.w)
        lin_vel = msg.twist.twist.linear
        # ang_vel = msg.twist.twist.angular
        self.vel1.append([lin_vel.x, lin_vel.y])
        self.odom1.append([posx, posy])
        roll, pitch, yaw = transform_orientation(orientation=orientation)
        self.yaw1.append(yaw)


    def odom_adv2_callback(self, msg):
            
        # message = msg
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        (posx, posy, posz) = (position.x, position.y, position.z)
        (qx, qy, qz, qw) = (orientation.x, orientation.y, orientation.z, orientation.w)
        lin_vel = msg.twist.twist.linear
        # ang_vel = msg.twist.twist.angular
        self.vel2.append([lin_vel.x, lin_vel.y])
        self.odom2.append([posx, posy])
        roll, pitch, yaw = transform_orientation(orientation=orientation)
        self.yaw2.append(yaw)


    def odom_adv3_callback(self, msg):
            
        # message = msg
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        (posx, posy, posz) = (position.x, position.y, position.z)
        (qx, qy, qz, qw) = (orientation.x, orientation.y, orientation.z, orientation.w)
        lin_vel = msg.twist.twist.linear
        # ang_vel = msg.twist.twist.angular
        self.vel3.append([lin_vel.x, lin_vel.y])
        self.odom3.append([posx, posy])
        roll, pitch, yaw = transform_orientation(orientation=orientation)
        self.yaw3.append(yaw)


    def scan_processor(self, msg):
        # currently returns closest quadrant, similar to odom, puts it into a queue
        ranges = msg.ranges
        intensities = msg.intensities
        # self.get_logger().info(f'ranges are: {ranges}, intensities are: {intensities}, lengths are: {len(ranges), len(intensities)}') # 360, 360
        
        quadrant_clip = 3
        closest_quadrant = 0
        # want one for each quadrant
        if self.quandrants:
            range_length_quad = int(len(ranges)/4)
            first_quadrant = [ranges[:range_length_quad*1], intensities[:range_length_quad*1]]
            second_quadrant = [ranges[range_length_quad*1:range_length_quad*2], intensities[range_length_quad*1:range_length_quad*2]]
            third_quadrant = [ranges[range_length_quad*2:range_length_quad*3], intensities[range_length_quad*2:range_length_quad*3]]
            fourth_quadrant = [ranges[range_length_quad*3:], intensities[range_length_quad*3:]]

            first_quadrant = np.array(first_quadrant[0])
            first_quadrant = np.clip(first_quadrant, 0, quadrant_clip)
            second_quadrant = np.array(second_quadrant[0])
            second_quadrant = np.clip(second_quadrant, 0, quadrant_clip)
            third_quadrant = np.array(third_quadrant[0])
            third_quadrant = np.clip(third_quadrant, 0, quadrant_clip)
            fourth_quadrant = np.array(fourth_quadrant[0])
            fourth_quadrant = np.clip(fourth_quadrant, 0, quadrant_clip)

            q_1_avg = sum(first_quadrant)/len(first_quadrant)
            q_2_avg = sum(second_quadrant)/len(second_quadrant)
            q_3_avg = sum(third_quadrant)/len(third_quadrant)
            q_4_avg = sum(fourth_quadrant)/len(fourth_quadrant)
            averages = [q_1_avg, q_2_avg, q_3_avg, q_4_avg]
            maxes = [sum(first_quadrant)]
            
            min_index = averages.index(min(averages))
            # self.get_logger().info(f'the maximum quadrant is number: {max_index}')

            closest_quadrant =  self.closest_quadrant([first_quadrant, second_quadrant, third_quadrant, fourth_quadrant])
            # self.get_logger().info(f'averages are: {averages, min_index}')
            # self.get_logger().info(f'closest quadrant is: {closest_quadrant}')

        else:

            range_length_oct = int(len(ranges)/8)
            ranges = np.array(ranges)
            front = [ranges[range_length_oct*1:range_length_oct*3], intensities[range_length_oct*1:range_length_oct*3]]
            right = [ranges[range_length_oct*3:range_length_oct*5], intensities[range_length_oct*3:range_length_oct*5]]
            rear = [ranges[range_length_oct*5:range_length_oct*7], intensities[range_length_oct*5:range_length_oct*7]]
            left_1 = [ranges[:range_length_oct*1], intensities[:range_length_oct*1]]
            left_2 = [ranges[:range_length_oct*1], intensities[:range_length_oct*1]]

            front = np.array(front[0])
            front = np.clip(front, 0, quadrant_clip)
            right = np.array(right[0])
            right = np.clip(right, 0, quadrant_clip)
            rear = np.array(rear[0])
            rear = np.clip(rear, 0, quadrant_clip)
            left_1 = np.array(left_1[0])
            left_1 = np.clip(left_1, 0, quadrant_clip)
            left_2 = np.array(left_2[0])
            left_2 = np.clip(left_2, 0, quadrant_clip)

            left = np.append(left_1, left_2)

            p_1_avg = sum(front)/len(front)
            p_2_avg = sum(right)/len(right)
            p_3_avg = sum(rear)/len(rear)
            p_4_avg = sum(left)/len(left)
            averages = [p_1_avg, p_2_avg, p_3_avg, p_4_avg]
            # maxes = [sum(first_quadrant[0])]
            
            min_index = averages.index(min(averages))

            # self.get_logger().info(f'averages are: {averages, min_index}')

            closest_quadrant =  self.closest_quadrant([front, right, rear, left])
            
            # self.get_logger().info(f'closest quadrant is: {closest_quadrant}')

        return closest_quadrant


    def scan_agent0_callback(self, msg):
        closest_quadrant = self.scan_processor(msg)
        self.closest_quadrant1 = closest_quadrant


    def scan_adv1_callback(self, msg):
        closest_quadrant = self.scan_processor(msg)
        self.closest_quadrant2 = closest_quadrant


    def scan_adv2_callback(self, msg):
        closest_quadrant = self.scan_processor(msg)
        self.closest_quadrant3 = closest_quadrant


    def scan_adv3_callback(self, msg):
        closest_quadrant = self.scan_processor(msg)
        self.closest_quadrant4 = closest_quadrant
        

    def quad_up_right(self, up, left, yaw):
        # we assume we're already in the right quadrant
        message = Twist()
        scale = abs(up)-abs(left) # between -1 and 1
        desired_yaw = (scale + 1) * math.pi/4 # now this is between 0 and pi/2
        yaw_delta = yaw - desired_yaw # assume yaw already between 0 and pi/2
        # if yaw_delta is positive, we turn right a bit, if negative, we turn left a bit
        message.linear.x = 1 * self.linear_vel_scale_factor
        message.angular.z = - (yaw_delta / (math.pi/2)) * 0.1 # as max yaw_delta here is between 0 and pi/2, we want max angular vel to be 0.2, we normalzie
        # add -1 sign because of rotation direction
        return message


    def quad_up_left(self, up, left, yaw):
        # we assume we're already in the right quadrant
        message = Twist()
        scale = abs(left) - abs(up)# between -1 and 1, need to swap left, up compard to up, right
        desired_yaw = (scale + 1) * math.pi/2 # now this is between pi/2 and pi
        yaw_delta = yaw - desired_yaw # assume yaw already between pi/2 and pi
        # if yaw_delta is positive, we turn right a bit, if negative, we turn left a bit
        message.linear.x = 1 * self.linear_vel_scale_factor
        message.angular.z = - (yaw_delta / (math.pi/2)) * 0.1 # as max yaw_delta here is between 0 and pi/2, we want max angular vel to be 0.2, we normalzie
        # add -1 sign because of rotation direction
        return message


    def quad_down_left(self, up, left, yaw):
        # we assume we're already in the right quadrant
        message = Twist()
        scale = abs(up) - abs(left)# between -1 and 1
        desired_yaw = - math.pi + ((scale + 1) * math.pi/4) # now this is between -pi and -pi/2
        yaw_delta = yaw - desired_yaw # assume yaw already between -pi and -pi/2
        # if yaw_delta is positive, we turn right a bit, if negative, we turn left a bit
        message.linear.x = 1 * self.linear_vel_scale_factor
        message.angular.z = - (yaw_delta / (math.pi/2)) * 0.1 # as max yaw_delta here is between 0 and pi/2, we want max angular vel to be 0.2, we normalzie
        # add -1 sign because of rotation direction
        return message


    def quad_down_right(self, up, left, yaw):
        # we assume we're already in the right quadrant
        message = Twist()
        scale = abs(left) - abs(up)# between -1 and 1
        desired_yaw = - math.pi/2 + ((scale + 1) * math.pi/4) # now this is between -pi and -pi/2
        yaw_delta = yaw - desired_yaw # assume yaw already between -pi and -pi/2
        # if yaw_delta is positive, we turn right a bit, if negative, we turn left a bit
        message.linear.x = 1 * self.linear_vel_scale_factor
        message.angular.z = - (yaw_delta / (math.pi/2)) * 0.1 # as max yaw_delta here is between 0 and pi/2, we want max angular vel to be 0.2, we normalzie
        # add -1 sign because of rotation direction
        return message


    def quad_control(self, action, yaw, agent): # not done yet
        # figure out more precise movements based on action
        # might be needed if simple control isnt enough
        if agent == 0:
            up =  action[3] - action[4]
            left =  action[2] - action[1]
                
        else:
            up =  action[4] - action[3]
            left =  action[1] - action[2]

        # agent wants to move to top-left quadrant
        if left>=0 and up>=0:
            if math.pi/2 <= yaw < math.pi:
                message = self.quad_up_left(up, left, yaw)
            else:
                if left>up:
                    message = self.go_left(yaw)
                    
                else:
                    message = self.go_up(yaw)
                    
        elif left>=0 and up<=0:
            if -math.pi <= yaw < -math.pi/2:
                message = self.quad_down_left(up, left, yaw)
            else:
                if abs(left) > abs(up):
                    message = self.go_left(yaw)
                    
                else:
                    message = self.go_down(yaw)
                    
        elif left<=0 and up>=0:
            if 0 <= yaw < math.pi/2:
                message = self.quad_up_right(up, left, yaw)
            else:
                if abs(left) > abs(up):
                    message = self.go_right(yaw)
                    
                else:
                    message = self.go_up(yaw)
                    
        elif left<=0 and up<=0:
            if -math.pi/2 <= yaw < 0:
                message = self.quad_down_right(up, left, yaw)
            else:    
                if abs(left) > abs(up):
                    message = self.go_right(yaw)
                    
                else:
                    message = self.go_down(yaw)
                    
        else:
            # message would be no action, or up-down or right-left = 0?
            message = self.dont_move()
        return message


    def go_down(self, yaw):
        message = Twist()
        
        # figure out if turn left or right
        difference = 2

        if abs(yaw)<math.pi/2:
            # want to turn left
            message.angular.z = difference*0.25 # some positive number, with some scaling factor
            message.linear.x = 1 * self.linear_vel_scale_factor # looks straight relative to robot
        else:
            #turn right
            message.angular.z = -difference*0.25 # some positive number, with some scaling factor
            message.linear.x = 1 * self.linear_vel_scale_factor # looks straight relative to robot
        return message
    

    def go_up(self, yaw):
        message = Twist()
        # figure out if turn left or right
        difference = 2

        if abs(yaw)>math.pi/2:
            # want to turn left
            message.angular.z = difference*0.25 # some positive number, with some scaling factor
            message.linear.x = 1 * self.linear_vel_scale_factor # looks straight relative to robot
        else:
            #turn right
            message.angular.z = -difference*0.25 # some positive number, with some scaling factor
            message.linear.x = 1 * self.linear_vel_scale_factor # looks straight relative to robot
        return message


    def go_left(self, yaw):
        message = Twist()
        # figure out if turn left or right
        difference = 2 # maybe change it for up n down?

        # if yaw positive, turl right, else left
        if yaw > 0:
            #turn right
            message.angular.z = -difference*0.25 # some positive number, with some scaling factor
            message.linear.x = 1 * self.linear_vel_scale_factor # looks straight relative to robot
        else:
            message.angular.z = difference*0.25 # some positive number, with some scaling factor
            message.linear.x = 1 * self.linear_vel_scale_factor # looks straight relative to robot
        return message


    def go_right(self, yaw):
        message = Twist()
        # figure out if turn left or right
        difference = 2 # maybe change it for up n down?

        # if yaw positive, turl right, else left
        if yaw < 0:
            #turn right
            message.angular.z = -difference*0.25 # some positive number, with some scaling factor
            message.linear.x = 1 * self.linear_vel_scale_factor # looks straight relative to robot
        else:
            message.angular.z = difference*0.25 # some positive number, with some scaling factor
            message.linear.x = 1 * self.linear_vel_scale_factor # looks straight relative to robot
        return message


    def dont_move(self):
        print(f'In the else dont_move message')
        message = Twist()
        message.linear.x = 0.
        message.linear.y = 0.
        message.linear.z = 0.
        message.angular.x = 0.
        message.angular.y = 0.
        message.angular.z = 0.
        return message


    def avoid_obstacle_move(self):
        print(f'trying to avoid an obstacle')
        message = Twist()
        message.linear.x = self.linear_vel_scale_factor * 0.5 # move at half speed
        message.linear.y = 0.
        message.linear.z = 0.
        message.angular.x = 0.
        message.angular.y = 0.
        message.angular.z = - 1.0 # turn right fast
        return message


    # returns the true closest quadrant, along with the shortest distance to an obstacle along that quadrant
    def true_closest_quadrant(self, quadrants):

        quadrant_mins = [min(elements) for elements in quadrants]
        closest_quadrant = quadrant_mins.index(min(quadrant_mins))
        return [closest_quadrant, min(quadrant_mins)]
    
    # however since we go straight, we only care about the frontal quadrant
    def closest_quadrant(self, quadrants):

        quadrant_mins = [min(elements) for elements in quadrants]
        closest_quadrant = quadrant_mins.index(min(quadrant_mins))
        return [closest_quadrant, quadrant_mins[3]] # return the frontal closest one, which quadrant is truly closest is not used


    def agents_closest_quadrant(self, agent):
        match agent:
            case 0:
                closest_quadrant = self.closest_quadrant1
                self.get_logger().info(f'closest quadrant for agent 0 is: {closest_quadrant}')
            case 1:
                closest_quadrant = self.closest_quadrant2
            case 2:
                closest_quadrant = self.closest_quadrant3
            case 3:
                closest_quadrant = self.closest_quadrant4
                
        return closest_quadrant


    def message_from_action(self, action, yaw, agent = 1):
        closest_quadrant = self.agents_closest_quadrant(agent)
        will_collide = True if closest_quadrant[1] <= self.collision_boundary else False


        if not will_collide:
            if agent == 0:
                up =  action[3] - action[4]
                left =  action[2] - action[1]
                
            else:
                up =  action[4] - action[3]
                left =  action[1] - action[2]

            if not self.do_quad_control:
                """
                # dont_move = action[0] # not relevant
                action = action[1:]
                max_action = action.argmax()
                # see which one is largest, move that way for now
                # def needs adjusting
                match max_action:
                    case 0:
                        # if is_agent: print(f'command is: go_left')
                        # if is_agent: return go_right(yaw)
                        # if is_adv1: print(f'command is: go_left')
                        message = go_left(yaw)
                    case 1:
                        # if is_agent: print(f'command is: go_right')
                        # if is_agent: return go_left(yaw)
                        # if is_adv1: print(f'command is: go_right')
                        message = go_right(yaw)
                    case 2:
                        # if is_agent: print(f'command is: go_down')
                        message = go_down(yaw)
                        # if is_adv1: print(f'command is: go_down')
                    case 3:
                        # if is_agent: print(f'command is: go_up')
                        message = go_up(yaw)
                        # if is_adv1: print(f'command is: go_up')
                    case "monke":
                        message = Twist()
                        message.linear.x = 0.
                        message.linear.y = 0.
                        message.linear.z = 0.
                        message.angular.x = 0.
                        message.angular.y = 0.
                        message.angular.z = 0.
                """
                if left>=0 and up>=0:
                        if left>up:
                            message = self.go_left(yaw)
                            # print('move left > up')
                        else:
                            message = self.go_up(yaw)
                            # print('move up > left')
                elif left>=0 and up<=0:
                        if abs(left) > abs(up):
                            message = self.go_left(yaw)
                            # print('move left > down')
                        else:
                            message = self.go_down(yaw)
                            # print('move down > left')
                elif left<=0 and up>=0:
                        if abs(left) > abs(up):
                            message = self.go_right(yaw)
                            # print('move right > up')
                        else:
                            message = self.go_up(yaw)
                            # print('move up > right')
                elif left<=0 and up<=0:
                        if abs(left) > abs(up):
                            message = self.go_right(yaw)
                            # print('move right > down')
                        else:
                            message = self.go_down(yaw)
                            # print('move down > right')
                else:
                    message = self.dont_move()
                    # message would be no action, or up-down or right-left = 0?
                return message
                    
            else:
                message = self.quad_control(action, yaw, agent=agent)

            return message
        
        elif will_collide:
            if closest_quadrant[1] <= self.stop_boundary:
                print(f'in obstacle avoidance, not moving')
                message = self.dont_move()
            # no obstackle avoidance implemented for quad control yet
            # elif self.do_quad_control:
                # message = self.quad_control(action, yaw, agent=agent)

            # elif not self.do_quad_control:
            # for now, just turn right if would hit obstacle

            else:
                 print(f'in obstacle avoidance, avoid obstacle move')
                 message = self.avoid_obstacle_move()

            return message


    def create_messages(self, actions):

        agent_0_action = actions["agent_0"]
        adversary_1_action = actions["adversary_0"]
        adversary_2_action = actions["adversary_1"]
        adversary_3_action = actions["adversary_2"]
        
        # from https://pettingzoo.farama.org/environments/mpe/simple_adversary/

        print_actions = False

        if print_actions:
            # index = np.argmax(agent_0_action)
            # up = agent_0_action[4] - agent_0_action[3]
            # left = agent_0_action[1] - agent_0_action[2]
            index = np.argmax(adversary_1_action)
            up =  adversary_1_action[3] - adversary_1_action[4]
            left =  adversary_1_action[2] - adversary_1_action[1]

            match index:
                case 0:
                    print("no_action")
                case 1:
                    print("move_right")
                case 2:
                    print("move_left")
                case 3:
                    print("move_up")
                case 4:
                    print("move_down")
            
            if left>0 and up>0:
                if left>up:
                    print('move left > up')
                else:
                    print('move up > left')
            if left>0 and up<0:
                if abs(left) > abs(up):
                    print('move left > down')
                else:
                    print('move down > left')
            if left<0 and up>0:
                if abs(left) > abs(up):
                    print('move right > up')
                else:
                    print('move up > right')
            if left<0 and up<0:
                if abs(left) > abs(up):
                    print('move right > down')
                else:
                    print('move down > right')

            # dist = distance(self.odom4[-1], self.odom1[-1])

        agent0_message = self.message_from_action(agent_0_action, self.yaw4.pop(), agent=0)
        adv1_message = self.message_from_action(adversary_1_action, self.yaw1.pop())
        adv2_message = self.message_from_action(adversary_2_action, self.yaw2.pop())
        adv3_message = self.message_from_action(adversary_3_action, self.yaw3.pop())


        messages = [agent0_message, adv1_message, adv2_message, adv3_message]
        return messages


    def timer_callback(self):
        # msg = String()
        # a_1_posx = self.odom4.pop()[0]
        # adv_1_posx = self.odom1.pop()[0]
        # adv_2_posx = self.odom2.pop()[0]
        # adv_3_posx = self.odom3.pop()[0]
        # msg.data = f'agent1, posx: {a_1_posx}'
        # self.dummy_pos_pub.publish(msg)

        obs = self.process_odom() # obs, distances

        # self.get_logger().info(f'agent1 posx: {a_1_posx}, adv1 posx: {adv_1_posx}, adv2 posx: {adv_2_posx}, adv3 posx: {adv_3_posx}')
        # self.get_logger().info(f'obs: {obs}') #, distances: {distances}')
        self.i += 1

        actions = self.call_network(obs)
        # agent_obs = obs["agent_0"]
        # adv1_obs = obs["adversary_0"]

        cmd_vel_messages = self.create_messages(actions)

        publish = True
        wipe_queues = False

        if publish:
            self.agent_cmd_vel_pub.publish(cmd_vel_messages[0]) 
            self.adv1_cmd_vel_pub.publish(cmd_vel_messages[1]) 
            self.adv2_cmd_vel_pub.publish(cmd_vel_messages[2]) 
            self.adv3_cmd_vel_pub.publish(cmd_vel_messages[3]) 
    
        if wipe_queues:
            if self.i > 10:
                self.odom1 = self.odom2 = self.odom3 = self.odom4 = [] # careful, afterwards empty list
                self.yaw1=self.yaw2=self.yaw3=self.yaw4 = []
                self.vel1 = self.vel2 = self.vel3 = self.vel14 = []
            self.i = 0
    

    def process_odom(self):
        adv_0_arr = [0]*12
        adv_0_xy = self.odom1.pop() # 12 len
        adv_1_arr = [0]*12
        adv_1_xy = self.odom2.pop()
        adv_2_arr = [0]*12
        adv_2_xy = self.odom3.pop()
        agent_0_arr = [0]*10
        ag_0_xy = self.odom4.pop()

        adv0_vel = self.vel1.pop()
        adv1_vel = self.vel2.pop()
        adv2_vel = self.vel3.pop()
        agent0_vel = self.vel4.pop()

        distances = calculate_relative_ditances(adv_0_xy, adv_1_xy, adv_2_xy, ag_0_xy)

        # doing agent 0
        # swap first two
        agent_0_arr[0] = agent0_vel[0] # self velX
        agent_0_arr[1] = agent0_vel[1] # self velY
        agent_0_arr[2] = ag_0_xy[0] # self posX
        agent_0_arr[3] = ag_0_xy[1] # self posY
        agent_0_arr[4] = distances[0] # agent 0 - adv 0 relative x pos
        agent_0_arr[5] = distances[1] # agent 0 - adv 0 relative y pos
        agent_0_arr[6] = distances[2] # agent 0 - adv 1 relative x pos
        agent_0_arr[7] = distances[3] # agent 0 - adv 1 relative y pos
        agent_0_arr[8] = distances[4] # agent 0 - adv 2 relative x pos
        agent_0_arr[9] = distances[5] # agent 0 - adv 2 relative y pos

        # adv 0
        adv_0_arr[0] = adv0_vel[0]      # self velX
        adv_0_arr[1] = adv0_vel[1]      # self velY
        adv_0_arr[2] = adv_0_xy[0]      # self posX
        adv_0_arr[3] = adv_0_xy[1]      # self posY
        adv_0_arr[4] = distances[6]     # adv 0 - adv 1 relative x pos
        adv_0_arr[5] = distances[7]     # adv 0 - adv 1 relative y pos
        adv_0_arr[6] = distances[8]     # adv 0 - adv 2 relative x pos
        adv_0_arr[7] = distances[9]     # adv 0 - adv 2 relative y pos
        adv_0_arr[8] = -1* distances[0] # adv 0 - agent 0 relative x pos
        adv_0_arr[9] = -1* distances[1] # adv 0 - agent 0 relative y pos
        adv_0_arr[10] = agent0_vel[0]   # agent 0 absolute vel
        adv_0_arr[11] = agent0_vel[1]   # agent 0 absolute vel

        # adv 1
        adv_1_arr[0] = adv1_vel[0] # self velX
        adv_1_arr[1] = adv1_vel[1] # self velY
        adv_1_arr[2] = adv_1_xy[0] # self posX
        adv_1_arr[3] = adv_1_xy[1] # self posY
        adv_1_arr[4] = -1* distances[6] # adv 1 - adv 0 relative x pos
        adv_1_arr[5] = -1* distances[7] # adv 1 - adv 0 relative y pos
        adv_1_arr[6] = distances[10] # adv 1 - adv 2 relative x pos
        adv_1_arr[7] = distances[11] # adv 1 - adv 2 relative y pos
        adv_1_arr[8] = -1* distances[2] # adv 1 - agent 0 relative x pos
        adv_1_arr[9] = -1* distances[3] # adv 1 - agent 0 relative y pos
        adv_1_arr[10] = agent0_vel[0]    # agent 0 absolute vel
        adv_1_arr[11] = agent0_vel[1]   # agent 0 absolute vel

        # adv 2
        adv_2_arr[0] = adv2_vel[0] # self velX
        adv_2_arr[1] = adv2_vel[1] # self velY
        adv_2_arr[2] = adv_2_xy[0] # self posX
        adv_2_arr[3] = adv_2_xy[1] # self posY
        adv_2_arr[4] = -1* distances[8] # adv 2 - adv 0 relative x pos
        adv_2_arr[5] = -1* distances[9] # adv 2 - adv 0 relative y pos
        adv_2_arr[6] = -1* distances[10] # adv 2 - adv 1 relative x pos
        adv_2_arr[7] = -1* distances[11] # adv 2 - adv 1 relative y pos
        adv_2_arr[8] = -1* distances[4] # adv 2 - agent 0 relative x pos
        adv_2_arr[9] = -1* distances[5] # adv 2 - agent 0 relative y pos
        adv_2_arr[10] = agent0_vel[0]    # agent 0 absolute vel
        adv_2_arr[11] = agent0_vel[1]    # agent 0 absolute vel

        obs = {"agent_0":agent_0_arr, "adversary_0":adv_0_arr, 
               "adversary_1":adv_1_arr, "adversary_2":adv_2_arr}

        return obs # , distances
     

def main(args=None):

    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()