import rclpy
from rclpy.node import Node

from std_msgs.msg import Int64
from tutorial_interfaces.msg import Num  
from geometry_msgs.msg import Twist
import math
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.cmd_vel_pub_ = self.create_publisher(Odometry, "/dummy_odom", 10)
        # self.timer_ = self.create_timer(0.5, self.send_velocity_command) # send command every 0.5s
        self.get_logger().info("Listening to odom node has been started")

    def odom_callback(self, msg):
        # message = Twist()
        message = msg
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        (posx, posy, posz) = (position.x, position.y, position.z)
        (qx, qy, qz, qw) = (orientation.x, orientation.y, orientation.z, orientation.w)
        # self.get_logger().info(f'posx: {posx}, posy: {posy}, posz: {posz}')


        self.cmd_vel_pub_.publish(message)   



    def closest_quadrant(self, quadrants):

        quadrant_mins = [min(elements) for elements in quadrants]
        closest_quadrant = quadrant_mins.index(min(quadrant_mins))

        return closest_quadrant


    def scan_callback(self, msg):
        ranges = msg.ranges
        intensities = msg.intensities
        # self.get_logger().info(f'ranges are: {ranges}, intensities are: {intensities}, lengths are: {len(ranges), len(intensities)}') # 360, 360


        quandrants = True
        quadrant_clip = 3
        # want one for each quadrant
        # Quadrants: 9-12h is first, then clockwise
        if quandrants:
            range_length_quad = int(len(ranges)/4)
            first_quadrant = [ranges[:range_length_quad*1], intensities[:range_length_quad*1]]
            second_quadrant = [ranges[range_length_quad*1:range_length_quad*2], intensities[range_length_quad*1:range_length_quad*2]]
            third_quadrant = [ranges[range_length_quad*2:range_length_quad*3], intensities[range_length_quad*2:range_length_quad*3]]
            fourth_quadrant = [ranges[range_length_quad*3:], intensities[range_length_quad*3:]]

            first_quadrant = numpy.array(first_quadrant[0])
            first_quadrant = numpy.clip(first_quadrant, 0, quadrant_clip)
            second_quadrant = numpy.array(second_quadrant[0])
            second_quadrant = numpy.clip(second_quadrant, 0, quadrant_clip)
            third_quadrant = numpy.array(third_quadrant[0])
            third_quadrant = numpy.clip(third_quadrant, 0, quadrant_clip)
            fourth_quadrant = numpy.array(fourth_quadrant[0])
            fourth_quadrant = numpy.clip(fourth_quadrant, 0, quadrant_clip)

            q_1_avg = sum(first_quadrant)/len(first_quadrant)
            q_2_avg = sum(second_quadrant)/len(second_quadrant)
            q_3_avg = sum(third_quadrant)/len(third_quadrant)
            q_4_avg = sum(fourth_quadrant)/len(fourth_quadrant)
            averages = [q_1_avg, q_2_avg, q_3_avg, q_4_avg]
            maxes = [sum(first_quadrant)]
            
            min_index = averages.index(min(averages))
            # self.get_logger().info(f'the maximum quadrant is number: {max_index}')

            closest_quadrant =  self.closest_quadrant([first_quadrant, second_quadrant, third_quadrant, fourth_quadrant])
            self.get_logger().info(f'averages are: {averages, min_index}')
            self.get_logger().info(f'closest quadrant is: {closest_quadrant}')

        else:

            range_length_oct = int(len(ranges)/8)
            ranges = numpy.array(ranges)
            front = [ranges[range_length_oct*1:range_length_oct*3], intensities[range_length_oct*1:range_length_oct*3]]
            right = [ranges[range_length_oct*3:range_length_oct*5], intensities[range_length_oct*3:range_length_oct*5]]
            rear = [ranges[range_length_oct*5:range_length_oct*7], intensities[range_length_oct*5:range_length_oct*7]]
            left_1 = [ranges[:range_length_oct*1], intensities[:range_length_oct*1]]
            left_2 = [ranges[:range_length_oct*1], intensities[:range_length_oct*1]]

            front = numpy.array(front[0])
            front = numpy.clip(front, 0, quadrant_clip)
            right = numpy.array(right[0])
            right = numpy.clip(right, 0, quadrant_clip)
            rear = numpy.array(rear[0])
            rear = numpy.clip(rear, 0, quadrant_clip)
            left_1 = numpy.array(left_1[0])
            left_1 = numpy.clip(left_1, 0, quadrant_clip)
            left_2 = numpy.array(left_2[0])
            left_2 = numpy.clip(left_2, 0, quadrant_clip)

            left = numpy.append(left_1, left_2)

            p_1_avg = sum(front)/len(front)
            p_2_avg = sum(right)/len(right)
            p_3_avg = sum(rear)/len(rear)
            p_4_avg = sum(left)/len(left)
            averages = [p_1_avg, p_2_avg, p_3_avg, p_4_avg]
            # maxes = [sum(first_quadrant[0])]
            
            min_index = averages.index(min(averages))

            # self.get_logger().info(f'averages are: {averages, min_index}')

            closest_quadrant =  self.closest_quadrant([front, right, rear, left])
            
            self.get_logger().info(f'closest quadrant is: {closest_quadrant}')
        # self.cmd_vel_pub_.publish(message) 


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