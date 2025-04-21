'''
this is the lander class
lander urdf: lander/urdf/lander.urdf
joint info:
(0, b'camera_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'camera_link', (0.0, 0.0, 0.0), (-0.8863800000000001, -2.3460099999999997, -0.31847), (0.0, 0.0, 0.0, 1.0), -1)
(1, b'leg1joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'leg1_link', (0.0, 0.0, 0.0), (-0.0034000000000000002, 2.89513, -2.07873), (0.0, 0.0, 0.0, 1.0), -1)
(2, b'leg2joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'leg2_link', (0.0, 0.0, 0.0), (-2.52361, 1.46176, -2.07873), (0.0, 0.0, 0.0, 1.0), -1)
(3, b'leg3joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'leg3_link', (0.0, 0.0, 0.0), (-2.52361, -1.4286, -2.07873), (0.0, 0.0, 0.0, 1.0), -1)
(4, b'leg4joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'leg4_link', (0.0, 0.0, 0.0), (-0.005920000000000002, -2.91426, -2.07873), (0.0, 0.0, 0.0, 1.0), -1)
(5, b'leg5joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'leg5_link', (0.0, 0.0, 0.0), (2.51176, -1.4739600000000002, -2.07873), (0.0, 0.0, 0.0, 1.0), -1)
(6, b'leg6joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'leg6_link', (0.0, 0.0, 0.0), (2.48908, 1.4179799999999998, -2.07873), (0.0, 0.0, 0.0, 1.0), -1)
(7, b'main_engine_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'engine_link', (0.0, 0.0, 0.0), (0.01235, -0.01476, -1.53869), (0.0, 0.0, 0.0, 1.0), -1)
(8, b'thruster1-joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'thruster1-link', (0.0, 0.0, 0.0), (1.88359, -0.01476, -0.75185), (0.0, 0.0, 0.0, 1.0), -1)
(9, b'thruster2-joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'thruster2-link', (0.0, 0.0, 0.0), (0.01235, 1.8791699999999998, -0.75185), (0.0, 0.0, 0.0, 1.0), -1)
(10, b'thruster4-joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'thruster4-link', (0.0, 0.0, 0.0), (0.01235, -1.89735, -0.75185), (0.0, 0.0, 0.0, 1.0), -1)
(11, b'thruster_3_joint', 4, -1, -1, 0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, b'thruster3_link', (0.0, 0.0, 0.0), (-1.88495, -0.01476, -0.70549), (0.0, 0.0, 0.0, 1.0), -1)

thruster1 and thruster3 are opposite to each other
thruster2 and thruster4 are opposite to each other


Mass Properties:
    - Total Launch Mass: 500 kg
    - Dry Mass: 200 kg
    - Propellant Mass: 300 kg (60% propellant fraction)

Engines:
    - Main Engine (center-mounted):
        • Thrust: 900–1100 N (Nominal: 1000 N)
        • Mass: 8.41 kg
        • Thrust-to-weight ratio on Moon: ~1.1–1.36
        • Specific Impulse (Isp): 300 s
        • Thrust-to-weight ratio on Earth: ~0.5–0.6

    - 4 Attitude Thrusters:
        • Thrust: 8–15 N each (total: 32–60 N)
        • Mass: 1.2 kg each (total: 4.8 kg)
        • Role: orientation and minor translational control
        • Specific Impulse (Isp): 220 s
        • Thrust-to-weight ratio on Moon: ~0.04–0.08
        • Thrust-to-weight ratio on Earth: ~0.08–0.16

Use Case:
    Designed for low-gravity landings (e.g., Moon, small bodies), with thrust and structural configuration suitable for soft landing, balance, and mid-course adjustments.
'''
import sys
from pathlib import Path

# A workaround for tests not automatically setting
# root/src/ as the current working directory
root_path = Path(__file__).parent.parent.parent

if root_path not in sys.path:
    sys.path.insert(0, str(root_path))

import pybullet as p
import pybullet_data
import numpy as np
import time
import tkinter as tk
from src.lander.telemetry_interface import TelemetryDisplay

MOON_GRAVITY = 1.62  # m/s^2
EARTH_GRAVITY = 9.81  # m/s^2


class Lander:
    def __init__(self, physics_engine, urdf_path=f"{root_path}/src/lander/urdf/lander.urdf", debug=False, telemetry=True):
        # Load the URDF file
        self.urdf_path = urdf_path
        self.lander_id = None
        self._p = physics_engine
        
        # thruster and engine
        self.debug = debug
        self.thrusters = [8, 9, 10, 11, 7]  # Joint indices for thrusters and main engine, -1 is for main engine
        self.main_engine = 7
        self.throttle = {7:0 , 8:0, 9:0, 10:0, 11:0}  # Throttle values for each thruster and engine
        self.thrusts = {7:0 , 8:0, 9:0, 10:0, 11:0}  # Thrust values for each thruster and engine
        self.Isp_engine = {7: 300, 8: 220, 9: 220, 10: 220, 11: 220}  # Specific impulse for each engine
        self.thrust_range = {7: (900, 1100), 8: (8, 15), 9: (8, 15), 10: (8, 15), 11: (8, 15)}  # Thrust range for each engine

        # sensor data
        self.initial_position = None
        self.initial_orientation = None
        self.initial_velocity = None
        self.initial_angular_velocity = None
        self.contact_state = {1: False, 2: False, 3: False, 4: False, 5: False, 6: False}  # Contact state for each leg
        self.current_mass = 0
        self.wet_mass = 0
        self.dry_mass = 60
        self.camera = None
        self.lidar_data = None
        self.imu_state = {
            "position": None,
            "orientation": None,
            "linear_velocity": None,
            "angular_velocity": None
        }
        self.avg_step_per_second = 240
        self.t_go = 0
        self.v_targ = 0

        #telemetry interface
        self.telemetry = None
        if telemetry:
            self.telemetry = TelemetryDisplay(tk.Tk())

    def load(self, position=(0, 0, 5), orientation=(0, 0, 0, 1), initial_velocity=(0, 0, 0), angular_velocity=(0, 0, 0)):
        """"
        Load the lander into the PyBullet simulation.
        Args:
            position (tuple): Initial position of the lander in the simulation.
            orientation (tuple): Initial orientation of the lander in quaternion format.
            initial_velocity (tuple): Initial linear velocity of the lander.
            angular_velocity (tuple): Initial angular velocity of the lander.
        """
        # Load the lander into the simulation
        self.lander_id = self._p.loadURDF(self.urdf_path, position, orientation)
        self.current_mass = self._p.getDynamicsInfo(self.lander_id, -1)[0]
        self.wet_mass = self.current_mass
        self._p.resetBaseVelocity(self.lander_id, linearVelocity=initial_velocity, angularVelocity=angular_velocity)
        self.initial_angular_velocity = angular_velocity
        self.initial_velocity = initial_velocity
        self.fuel_level = 100 * (self.current_mass - self.dry_mass) / (self.wet_mass - self.dry_mass)
        self.initial_position = position
        self.initial_orientation = orientation
        self.update_sensors()

        if self.telemetry:
            self.update_telemetry()

            
    def calculate_tgo_and_v(self, t1=20, t2=100):
        # 1) Read current state
        r = self.imu_state["position"]         # 3‑vector: [x, y, z] position
        v = self.imu_state["linear_velocity"]  # 3‑vector: [vx, vy, vz] velocity

        # 2) Compute a “modified” position error r̂
        #    If we’re above 15 m altitude, we care about how far we are above 15 m
        #    Otherwise we care only about current altitude (z), ignoring horizontal offset
        if r[2] > 15:
            r_hat = r - np.array([0, 0, 15])
        else:
            r_hat = np.array([0, 0, r[2]])

        # 3) Compute a “modified” velocity error v̂
        #    Above 15 m we subtract a -2 m/s downward bias; below 15 m, a -1 m/s bias
        #    (these offset values come from the guidance design)
        if r[2] > 15:
            v_hat = v - np.array([0, 0, -2])
        else:
            v_hat = v - np.array([0, 0, -1])

        # 4) Choose a time‑constant τ based on altitude
        tau = t1 if r[2] > 15 else t2

        # 5) Compute time‑to‑go (t_go) as distance over speed
        #    ‖r̂‖ / ‖v̂‖ is a crude estimate of how long it would take to close r̂
        self.t_go = np.linalg.norm(r_hat) / (np.linalg.norm(v_hat) + 1e-10)

        # 6) Compute the target velocity vector v_targ - {We might be able to use different way to calculate to get more fuel efficient trajectory}
        #    - Start with the initial descent speed magnitude (‖initial_velocity‖)
        #    - Point it in the direction of r̂ (so we “head back” toward the target)
        #    - Scale by (1 − exp(−t_go/τ)) so that early on we allow larger speed,
        #      and as t_go → 0, the term → 0, enforcing zero final velocity
        direction = r_hat / np.linalg.norm(r_hat)
        mag    = np.linalg.norm(self.initial_velocity)
        shape  = (1 - np.exp(-self.t_go / tau))
        self.v_targ = - mag * direction * shape

        
    def apply_thrust(self):
        '''
        Applies thrust to the lander by calculating and applying forces to its thrusters.
        This method iterates through all the thrusters of the lander, calculates the force 
        based on the throttle value for each thruster, and applies the force using the 
        PyBullet physics engine. Additionally, it applies a separate force for the main 
        engine using its throttle value.
        The forces are applied in the local frame of reference of the respective thrusters.
        Attributes:
            thrusters (list): A list of IDs representing the thrusters of the lander.
            throttle (dict): A dictionary mapping thruster IDs to their respective throttle values.
            lander_id (int): The unique ID of the lander object in the PyBullet simulation.
            main_engine (int): The ID of the main engine thruster.
        '''
        if self.current_mass == self.dry_mass:
            # If the lander is out of fuel, set throttle to 0
            for thruster_id in self.thrusters:
                self.throttle[thruster_id] = 0
            return
        
        for thruster_id in self.thrusters:
            force = [0,0, self.thrust_range[thruster_id][0] + (self.throttle[thruster_id] * (self.thrust_range[thruster_id][1] - self.thrust_range[thruster_id][0]))]
            if self.throttle[thruster_id] == 0:
                force = [0,0,0]

            self.thrusts[thruster_id] = force

            self._p.applyExternalForce(
                objectUniqueId=self.lander_id,
                linkIndex=thruster_id,
                forceObj=force,
                posObj=[0,0,0],
                flags=p.LINK_FRAME
            )
        
    def update_fuel(self):
        '''
        Here we implement a linear fuel consumption model based on the throttle values
        '''
        for thruster_id, throttle in self.throttle.items():
            if self.throttle[thruster_id] == 0:
                thrust = 0
            else:
                thrust = self.thrust_range[thruster_id][0] + (throttle * (self.thrust_range[thruster_id][1] - self.thrust_range[thruster_id][0]))
            burn_rate = (thrust) / (self.Isp_engine[thruster_id] * MOON_GRAVITY) # kg/s, assuming 1.8 m/s^2 gravity
            self.current_mass -= burn_rate * (1 / self.avg_step_per_second)  # Assuming 240 simulation steps per second
            self.current_mass = max(self.current_mass, self.dry_mass)  # Ensure mass doesn't go below dry mass
            self.fuel_level = 100 * (self.current_mass - self.dry_mass) / (self.wet_mass - self.dry_mass)
            # Update the mass of the base body to the new mass
            self._p.changeDynamics(self.lander_id, -1, mass=self.current_mass)

    def step(self, throttles):
        # Update the throttle values
        for thruster_id, throttle in throttles.items():
            self.throttle[thruster_id] = throttle
            
        if not hasattr(self, "step_times"):
            self.step_times = []
        current_time = time.time()
        if hasattr(self, "last_update_time"):
            dt = current_time - self.last_update_time
            self.step_times.append(dt)
            if len(self.step_times) > 60:
                self.step_times.pop(0)
            avg_dt = sum(self.step_times) / len(self.step_times)
            self.avg_step_per_second = 1.0 / avg_dt if avg_dt > 0 else 0
        self.last_update_time = current_time
        # print(self.avg_step_per_second)

        self.apply_thrust()
        self.update_fuel()
        self.update_sensors()
        self.calculate_tgo_and_v()
        if self.telemetry:
            self.update_telemetry()
            self.telemetry.update_telemetry()

    def get_camera_data(self):
        '''
        Get the RGBD camera data from the lander and display it in the debug output of PyBullet.
        The camera sensor is attached to link 0 (camera_link).
        The camera orientation is adjusted to be normal to link_state[1] with the Y-axis within the link frame.
        '''
        camera_link_index = 0  # Link index for the camera
        near_plane = 0.01
        far_plane = 100
        fov = 60
        aspect_ratio = 1.0

        # Get the camera's position and orientation
        link_state = self._p.getLinkState(self.lander_id, camera_link_index)
        camera_position = link_state[0]
        camera_orientation = link_state[1]

        # Adjust the camera orientation to be normal to link_state[1] with the Y-axis within the link frame
        rotation_matrix = self._p.getMatrixFromQuaternion(camera_orientation)
        forward_vector = [rotation_matrix[2], rotation_matrix[5], rotation_matrix[8]]  # Z-axis in link frame
        up_vector = [rotation_matrix[1], rotation_matrix[4], rotation_matrix[7]]  # Y-axis in link frame
        target_position = [camera_position[i] - forward_vector[i] for i in range(3)]

        # Debug: Add a line in the PyBullet GUI to show the camera direction
        if self.debug:
            self._p.addUserDebugLine(camera_position, target_position, [1, 0, 0], 2, 0.1)

        # Compute the projection matrix
        projection_matrix = self._p.computeProjectionMatrixFOV(fov, aspect_ratio, near_plane, far_plane)

        # Get the camera image
        width, height, rgb_image, _, _ = self._p.getCameraImage(
            width=64,
            height=64,
            viewMatrix=p.computeViewMatrix(camera_position, target_position, up_vector),
            projectionMatrix=projection_matrix
        )

        return rgb_image

    def update_sensors(self):
        """
        Update the sensor data for the lander.
        This method retrieves the camera data and Lidar data from the lander.
        """
        # self.camera = self.get_camera_data()
        # self.lidar_data = self.get_lidar_data()
        self.contact_state = self.get_contact_state()
        self.imu_state = self.get_position_and_velocity()

    def get_lidar_data(self):
        """
        Get Lidar data from the lander.
        The Lidar is located on link 0 (camera_link) and emits rays in a cone pattern.
        Returns:
        list: A list of distances for each ray cast by the Lidar.
        """
        lidar_link_index = 0  # Link index for the Lidar
        num_rays = 36  # Number of rays in the cone
        ray_length = 50  # Maximum length of each ray
        ray_start = self._p.getLinkState(self.lander_id, lidar_link_index)[0]  # Lidar position
        ray_directions = []

        # Generate ray directions in a cone pattern angled downward
        for i in range(num_rays):
            angle = (2 * np.pi / num_rays) * i
            ray_directions.append([
            ray_length * np.cos(angle),  # X direction
            ray_length * np.sin(angle),  # Y direction
            -ray_length * 0.5  # Z direction (angled downward)
            ])

        # Cast rays and collect results
        ray_end_positions = [[ray_start[0] + d[0], ray_start[1] + d[1], ray_start[2] + d[2]] for d in ray_directions]
        ray_results = self._p.rayTestBatch([ray_start] * num_rays, ray_end_positions)

        # Extract distances from ray results and add debug lines
        distances = []
        for i, result in enumerate(ray_results):
            distance = result[2] * ray_length if result[0] != -1 else ray_length
            distances.append(distance)

            # Add a debug line for each ray
            hit_position = ray_end_positions[i] if result[0] == -1 else result[3]
            if self.debug:
                self._p.addUserDebugLine(ray_start, hit_position, [0, 1, 0] if result[0] != -1 else [1, 0, 0], 1)

        return distances

    def get_contact_state(self):
        # Get the contact state of the legs
        for leg in self.contact_state.keys():
            contacts = self._p.getContactPoints(bodyA=self.lander_id, linkIndexA=leg)
            self.contact_state[leg] = len(contacts) > 0
        return self.contact_state

    def get_position_and_velocity(self):
        # Get the position, orientation, linear velocity, and angular velocity of the lander
        position, orientation = self._p.getBasePositionAndOrientation(self.lander_id)
        linear_velocity, angular_velocity = self._p.getBaseVelocity(self.lander_id)
        self.imu_state["position"] = np.array(position)
        self.imu_state["orientation"] = np.array(orientation)
        self.imu_state["linear_velocity"] = np.array(linear_velocity)
        self.imu_state["angular_velocity"] = np.array(angular_velocity)
        return self.imu_state
    
    def update_telemetry(self):
        '''
        Update the self.telemetry with the current state of the lander.

        Attributes in TelemetryInterface:
        - imu_state: {"pitch": 0.0, "roll": 0.0, "position": [0, 0, 0]} (in degrees)
        - fuel_level: 100.0 (percent)
        - lidar: [0.0] * 6 (Simulated LIDAR values for 6 directions)
        - throttles: {7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}
        - leg_contacts: [False, False, False, False, False, False]
        - time_step: 0
        - vertical_velocity: 0.0 (m/s)
        - horizontal_speed: 0.0 (m/s)
        - time_to_go: 0.0 (s)
        - engine_temperature: 25.0 (°C)
        - position_xy: (0.0, 0.0) (Simulated horizontal position relative to target)
        - position_history: [(0, 0)] (history for trajectory)
        '''
        self.telemetry.update_time.append(time.time())

        # Update IMU state
        imu_state = self.imu_state
        if imu_state["orientation"] is not None:
            roll, pitch, _ = self._p.getEulerFromQuaternion(imu_state["orientation"])
            self.telemetry.imu_state["roll"] = np.degrees(roll)
            self.telemetry.imu_state["pitch"] = np.degrees(pitch)
        self.telemetry.imu_state["position"] = imu_state["position"]

        # Update fuel level (placeholder logic, adjust as needed)
        self.telemetry.fuel_level = self.fuel_level

        # Update LIDAR data
        # self.telemetry.lidar = self.lidar_data if self.lidar_data else [0.0] * 6

        # Update throttles
        self.telemetry.throttles = self.throttle

        # Update leg contact states
        self.telemetry.leg_contacts = self.contact_state

        # Update vertical velocity
        self.telemetry.vertical_velocity = imu_state["linear_velocity"][2] if (imu_state["linear_velocity"] is not None) else 0.0

        # Update horizontal speed
        if imu_state["linear_velocity"] is not None:
            vx, vy = imu_state["linear_velocity"][:2]
            self.telemetry.horizontal_speed = np.sqrt(vx**2 + vy**2)

        # Update position relative to target
        if imu_state["position"] is not None:
            self.telemetry.position_xy = (imu_state["position"][0], imu_state["position"][1])
            self.telemetry.position_history.append(self.telemetry.position_xy)
            self.telemetry.altitude_history.append(imu_state["position"][2])

        # Update engine temperature (placeholder logic, adjust as needed)
        self.telemetry.engine_temperature += 0.01 * sum(self.throttle.values())
        self.telemetry.engine_temperature = min(100.0, self.telemetry.engine_temperature)  # Cap at 100°C

        # Update time step
        self.telemetry.time_step += 1

# Example usage with sliders for each thruster
if __name__ == "__main__":
    # Initialize PyBullet self._p.DIRECT or self._p.GUI
    p.connect(p.GUI, options="--background_color_red=0 --background_color_blue=0 --background_color_green=0")
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -MOON_GRAVITY)

    # Set the background color to black (space-like environment)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

    # Load a plane as the grounda
    plane_id = p.loadURDF("plane.urdf", (0, 0, 0), useFixedBase=True)
    # p.loadURDF("planetary_surface.urdf", (0, 0, -10), useFixedBase=True)

    # Create and load the lander
    lander = Lander(p ,debug=False)
    lander.load()

    # Create sliders for each thruster and main engine
    sliders = {}
    for thruster_id in lander.thrusters:
        sliders[thruster_id] = p.addUserDebugParameter(f"Throttle {thruster_id}", 0, 1, 0)
    # global_slider = p.addUserDebugParameter("Global Throttle", 0, 30, 0)

    slider_x = p.addUserDebugParameter("Target X", -10, 10, 0)
    slider_y = p.addUserDebugParameter("Target Y", -10, 10, 0)
    slider_z = p.addUserDebugParameter("Target Z", 0, 10, 0)
    # Simulate
    while True:        
        throttles = {}

        # Read the target position from sliders
        target_z = p.readUserDebugParameter(slider_z)
        target_x = p.readUserDebugParameter(slider_x)
        target_y = p.readUserDebugParameter(slider_y)

        # Add a small sphere to visualize the user-defined position
        if not hasattr(lander, "target_visual"):
            sphere_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=1,
            rgbaColor=[1, 0, 0, 1]  # Red color
            )
            lander.target_visual = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=sphere_visual,
            basePosition=[target_x, target_y, target_z]
            )
        else:
            p.resetBasePositionAndOrientation(
            lander.target_visual,
            [target_x, target_y, target_z],
            [0, 0, 0, 1]
            )

        # Simple PID controller for position and orientation control
        Kp_altitude = 0.3  # Proportional gain for altitude
        Ki_altitude = 0.00008  # Integral gain for altitude
        Kd_altitude = 2.4  # Derivative gain for altitude

        Kp_orientation = 0.5  # Proportional gain for orientation
        Ki_orientation = 0  # Integral gain for orientation
        Kd_orientation = 0  # Derivative gain for orientation

        # Initialize PID variables
        if not hasattr(lander, "pid_error_sum"):
            lander.pid_error_sum = {"altitude": 0, "roll": 0, "pitch": 0}
            lander.pid_last_error = {"altitude": 0, "roll": 0, "pitch": 0}

        # Calculate current altitude and orientation from IMU data
        imu_state = lander.imu_state
        current_altitude = imu_state["position"][2] if imu_state["position"] else 0
        current_orientation = imu_state["orientation"] if imu_state["orientation"] else [0, 0, 0, 1]
        roll, pitch, _ = p.getEulerFromQuaternion(current_orientation)

        # Calculate PID terms for altitude
        error_altitude = target_z - current_altitude
        lander.pid_error_sum["altitude"] += error_altitude
        derivative_altitude = error_altitude - lander.pid_last_error["altitude"]
        lander.pid_last_error["altitude"] = error_altitude

        # Calculate PID terms for roll and pitch
        target_roll = 0  # Keep the lander level
        target_pitch = 0  # Keep the lander level
        error_roll = target_roll - roll
        error_pitch = target_pitch - pitch

        lander.pid_error_sum["roll"] += error_roll
        lander.pid_error_sum["pitch"] += error_pitch

        derivative_roll = error_roll - lander.pid_last_error["roll"]
        derivative_pitch = error_pitch - lander.pid_last_error["pitch"]

        lander.pid_last_error["roll"] = error_roll
        lander.pid_last_error["pitch"] = error_pitch

        # Compute throttle adjustments
        throttle_main_engine = (
            Kp_altitude * error_altitude +
            Ki_altitude * lander.pid_error_sum["altitude"] +
            Kd_altitude * derivative_altitude
        )
        throttle_thruster_1 = (
            Kp_orientation * -error_pitch +
            Ki_orientation * -lander.pid_error_sum["roll"] +
            Kd_orientation * -derivative_pitch
        )  # Thruster 1 for pitch
        throttle_thruster_3 = (
            Kp_orientation * error_pitch +
            Ki_orientation * lander.pid_error_sum["roll"] +
            Kd_orientation * derivative_pitch
        )  # Thruster 3 for pitch
        throttle_thruster_2 = (
            Kp_orientation * error_roll +
            Ki_orientation * lander.pid_error_sum["pitch"] +
            Kd_orientation * derivative_roll
        )  # Thruster 2 for roll
        throttle_thruster_4 = (
            Kp_orientation * -error_roll +
            Ki_orientation * -lander.pid_error_sum["pitch"] +
            Kd_orientation * -derivative_roll
        )  # Thruster 4 for roll

        # Apply throttle to the main engine (clamp between 0 and 1)
        throttles[lander.main_engine] = max(0, min(1, throttle_main_engine))

        # Apply throttle to each thruster separately
        throttles[8] = max(0, min(1, throttle_thruster_1))  # Thruster 1
        throttles[11] = max(0, min(1, throttle_thruster_3))  # Thruster 3
        throttles[9] = max(0, min(1, throttle_thruster_2))  # Thruster 2
        throttles[10] = max(0, min(1, throttle_thruster_4))  # Thruster 4

        # Apply thrust and update exhaust visualization
        next_state = lander.step(throttles)

        # camera follow lander
        # p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=imu_state["position"])

        # Step simulation
        # time.sleep(0.001)
        p.stepSimulation()
