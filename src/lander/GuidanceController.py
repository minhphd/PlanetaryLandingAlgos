import numpy as np

class GuidanceController:
    def __init__(self, lander, desired_position, desired_velocity, gravity):
        """
        Initialize the guidance controller.

        Parameters:
        - lander: Instance of your Lander class.
        - desired_position: Target landing position (e.g., [x, y, z]).
        - desired_velocity: Target landing velocity (typically [0, 0, 0] for soft landing).
        - gravity: Gravity vector (e.g., [0, 0, -9.81]).
        """
        self.lander = lander
        self.desired_position = np.array(desired_position)
        self.desired_velocity = np.array(desired_velocity)
        self.gravity = np.array(gravity)

    def estimate_state(self):
        """
        Estimate the current state of the lander using sensor data.
        
        Returns:
        - position: Current position as a 3D vector.
        - velocity: Current linear velocity as a 3D vector.
        """
        # Retrieve IMU data, etc. from the lander
        state = self.lander.get_position_and_velocity()
        position = np.array(state["position"])
        velocity = np.array(state["linear_velocity"])
        return position, velocity

    def compute_time_to_go(self):
        """
        Compute the time-to-go (t_go) until landing.
        This can be a fixed value or computed dynamically based on current altitude and descent rate.

        Returns:
        - t_go: Estimated remaining time until landing.
        """
        position, velocity = self.estimate_state()
        altitude = position[2]  # assuming Z is vertical
        vertical_velocity = velocity[2]
        # Simple estimation: time = altitude / descent rate (ensure vertical_velocity is non-zero)
        t_go = altitude / abs(vertical_velocity) if abs(vertical_velocity) > 1e-3 else 10.0
        return t_go

    def compute_zem(self, position, velocity, t_go):
        """
        Compute the Zero-Effort Miss (ZEM).

        ZEM represents the predicted position error at landing if no further thrust is applied.

        Formula:
          ZEM = desired_position - [position + t_go * velocity + 0.5 * gravity * t_go^2]

        Returns:
        - zem: A 3D vector representing the position error.
        """
        zem = self.desired_position - (position + t_go * velocity + 0.5 * self.gravity * t_go**2)
        return zem

    def compute_zev(self, velocity, t_go):
        """
        Compute the Zero-Effort Velocity (ZEV).

        ZEV represents the predicted velocity error at landing if no further thrust is applied.

        Formula:
          ZEV = desired_velocity - [velocity + gravity * t_go]

        Returns:
        - zev: A 3D vector representing the velocity error.
        """
        zev = self.desired_velocity - (velocity + self.gravity * t_go)
        return zev

    def compute_desired_acceleration(self, zem, zev, t_go):
        """
        Compute the desired acceleration based on ZEM and ZEV using the CTVG guidance law.

        Guidance law:
          a_desired = (6/t_go^2) * ZEM - (2/t_go) * ZEV

        Returns:
        - a_desired: A 3D acceleration vector.
        """
        a_desired = (6 / t_go**2) * zem - (2 / t_go) * zev
        return a_desired

    def thrust_allocation(self, a_desired):
        """
        Allocate the computed desired acceleration to the available thrusters.

        This method maps the desired acceleration to throttle commands for the main engine
        and the side thrusters, subject to their maximum force capabilities.
        
        Returns:
        - throttles: A dictionary mapping engine/thruster IDs to throttle commands.
        """
        # Example: Compute required force from F = m * a_desired.
        # Here we assume a simplified allocation focusing on the main engine.
        mass = self.lander.current_mass
        required_force = mass * np.linalg.norm(a_desired)
        
        # Assume maximum thrust for the main engine is known (e.g., 1000 units)
        max_main_thrust = 1000
        throttle_main = min(required_force / max_main_thrust, 1.0)
        
        # Additional logic can be added here to distribute thrust among side thrusters for attitude control.
        throttles = {
            self.lander.main_engine: throttle_main,
            # Optionally, add side thruster commands if needed.
        }
        return throttles

    def control_loop(self):
        """
        Main control loop to be executed at each simulation/control timestep.
        This integrates state estimation, ZEM/ZEV computation, desired acceleration calculation,
        thrust allocation, and finally commanding the lander.
        """
        # 1. Estimate current state
        position, velocity = self.estimate_state()
        
        # 2. Compute time-to-go
        t_go = self.compute_time_to_go()
        
        # 3. Compute ZEM and ZEV
        zem = self.compute_zem(position, velocity, t_go)
        zev = self.compute_zev(velocity, t_go)
        
        # 4. Compute desired acceleration from the guidance law
        a_desired = self.compute_desired_acceleration(zem, zev, t_go)
        
        # 5. Allocate thrust commands to engines
        throttles = self.thrust_allocation(a_desired)
        
        # 6. Apply thrust commands to the lander
        self.lander.step(throttles)
