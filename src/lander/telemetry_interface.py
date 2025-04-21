#!/usr/bin/env python3
import tkinter as tk
import math

class TelemetryDisplay:
    def __init__(self, root):
        self.root = root
        self.root.title("Lander Telemetry Display")
        self.canvas_width = 1200
        self.canvas_height = 800
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg="#1e1e1e")
        self.canvas.pack()

        # New: Create a Frame widget for the top-down trajectory graph.
        self.traj_frame = tk.Frame(root, bg="#1e1e1e")
        self.traj_frame.pack(fill=tk.BOTH, expand=True)
        self.traj_canvas = tk.Canvas(self.traj_frame, bg="#1e1e1e", height=300)
        self.traj_canvas.pack(fill=tk.BOTH, expand=True)

        # Initialize sensor values
        self.imu_state = {"pitch": 0.0, "roll": 0.0}  # in degrees
        self.fuel_level = 100.0                     # percent
        self.lidar = [0.0] * 6                      # Simulated LIDAR values for 6 directions
        self.throttles = {7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0}
        self.leg_contacts = {1: False, 2: False, 3: False, 4: False, 5: False, 6: False}

        self.time_step = 0
        
        # Additional telemetry variables
        self.vertical_velocity = 0.0  # m/s
        self.horizontal_speed = 0.0   # m/s
        self.time_to_go = 0.0         # s
        self.engine_temperature = 25.0  # °C
        # Simulated horizontal position (x,y) in meters relative to target at (0,0)
        self.position_history = []  # history for trajectory
        self.altitude_history = []  # history for altitude graph
        self.update_time = []
        
        self.update_telemetry()

    def close(self):
        self.root.destroy()
        self.root.quit()

# def draw_clock_and_eta(self):
#     x0, y0 = 50, 10
#     # Display the current time
#     current_time = self.root.after(0, lambda: self.root.winfo_toplevel().tk.call('clock', 'format', 'now', '-format', '%H:%M:%S'))
#     clock_text = f"Time: {current_time}"
#     self.canvas.create_text(x0, y0, anchor="nw", text=clock_text, fill="#ffffff", font=("Helvetica", 14, "bold"))

#     # Display the ETA
#     eta_text = f"ETA: {self.time_to_go:6.2f} s"
#     self.canvas.create_text(x0, y0 + 20, anchor="nw", text=eta_text, fill="#ffffff", font=("Helvetica", 14, "bold"))


    def draw_altitude_graph(self):
        x0, y0, width, height = 50, 400, 550, 200  # Adjusted width for side-by-side layout
        self.canvas.create_rectangle(x0, y0, x0 + width, y0 + height, outline="#ffffff", width=2)

        if len(self.altitude_history) > 1 and len(self.update_time) > 1:
            # Normalize time to fit within the graph width
            min_time = self.update_time[0]
            max_time = self.update_time[-1]
            time_range = max_time - min_time if max_time > min_time else 1
            scaled_times = [(t - min_time) / time_range * width for t in self.update_time]

            # Normalize altitude to fit within the graph height
            max_altitude = max(self.altitude_history) if self.altitude_history else 1
            scaled_altitudes = [min(height, (h / max_altitude) * height) for h in self.altitude_history]

            # Plot the altitude graph
            for i in range(1, len(scaled_altitudes)):
                x1 = x0 + scaled_times[i - 1]
                y1 = y0 + height - scaled_altitudes[i - 1]
                x2 = x0 + scaled_times[i]
                y2 = y0 + height - scaled_altitudes[i]
                self.canvas.create_line(x1, y1, x2, y2, fill="#00ffff", width=2)

        # Display the latest altitude value
        if self.altitude_history:
            altitude_text = "Altitude: {:.2f} m".format(self.altitude_history[-1])
            self.canvas.create_text(x0 + width / 2, y0 - 10, text=altitude_text, fill="#ffffff", font=("Helvetica", 14, "bold"))

    def draw_topdown_trajectory(self):
        x0, y0, size = 650, 400, 200  # Adjusted position and size for square layout
        self.canvas.create_rectangle(x0, y0, x0 + size, y0 + size, outline="#ffffff", width=2)

        if len(self.position_history) < 2:
            return

        # Target location is at the center
        target_x = x0 + size / 2
        target_y = y0 + size / 2
        self.canvas.create_oval(target_x - 5, target_y - 5, target_x + 5, target_y + 5, fill="#ff0000", outline="#ffffff")
        self.canvas.create_text(target_x, target_y - 10, text="TARGET", fill="#ffffff", font=("Helvetica", 10, "bold"))

        # Scaling: assume positions in meters; choose scale to fit trajectory history.
        scale = 0.1  # pixels per meter (adjust as needed)
        # Draw trajectory history as a polyline:
        trajectory_points = []
        for pos in self.position_history:
            px = target_x + pos[0] * scale
            py = target_y - pos[1] * scale
            trajectory_points.extend([px, py])
        self.canvas.create_line(trajectory_points, fill="#00ffff", width=2)

        # Draw current position marker
        cur_x = target_x + self.position_history[-1][0] * scale
        cur_y = target_y - self.position_history[-1][1] * scale
        self.canvas.create_oval(cur_x - 6, cur_y - 6, cur_x + 6, cur_y + 6, fill="#00ff00", outline="#ffffff", width=2)
        self.canvas.create_text(cur_x, cur_y - 10, text="CURRENT", fill="#ffffff", font=("Helvetica", 10, "bold"))

    def draw_artificial_horizon(self):
        x0, y0, width, height = 50, 20, 500, 200
        self.canvas.create_rectangle(x0, y0, x0 + width, y0 + height, outline="#ffffff", width=2)
        center_x = x0 + width / 2
        center_y = y0 + height / 2
        pitch = self.imu_state["pitch"]
        roll = self.imu_state["roll"]
        yaw = self.imu_state.get("yaw", 0.0)  # Default yaw to 0 if not provided

        radius = min(width, height) / 2 - 10

        # # Draw sky and ground
        # self.canvas.create_arc(center_x - radius, center_y - radius,
        #                        center_x + radius, center_y + radius,
        #                        start=0, extent=180, fill="#87CEEB", outline="")  # Sky
        # self.canvas.create_arc(center_x - radius, center_y - radius,
        #                        center_x + radius, center_y + radius,
        #                        start=180, extent=180, fill="#8B4513", outline="")  # Ground

        # Draw the circle boundary
        self.canvas.create_oval(center_x - radius, center_y - radius,
                                center_x + radius, center_y + radius,
                                outline="#ffffff", width=2)

        # Calculate roll and pitch adjustments
        roll_radians = math.radians(roll)
        pitch_offset = (pitch / 20.0) * radius

        # Draw the horizon line constrained within the circle
        line_length = radius * 1.5
        x1 = center_x - line_length * math.cos(roll_radians)
        y1 = center_y - line_length * math.sin(roll_radians) + pitch_offset
        x2 = center_x + line_length * math.cos(roll_radians)
        y2 = center_y + line_length * math.sin(roll_radians) + pitch_offset

        # Clip the horizon line to the circle
        def clip_to_circle(x, y):
            dx, dy = x - center_x, y - center_y
            dist = math.sqrt(dx**2 + dy**2)
            if dist > radius:
                scale = radius / dist
                return center_x + dx * scale, center_y + dy * scale
            return x, y

        x1, y1 = clip_to_circle(x1, y1)
        x2, y2 = clip_to_circle(x2, y2)
        self.canvas.create_line(x1, y1, x2, y2, fill="#00aaff", width=3)

        # Draw pitch markers
        for angle in range(-20, 21, 10):
            offset = (angle / 20.0) * radius
            marker_y = center_y + offset
            if -radius <= offset <= radius:
                self.canvas.create_line(center_x - 10, marker_y, center_x + 10, marker_y, fill="#ffffff", width=2)
                self.canvas.create_text(center_x + 20, marker_y, text=f"{angle:+}", fill="#ffffff", font=("Helvetica", 10, "bold"))

        # Draw roll markers
        for angle in range(-90, 91, 30):
            roll_angle_radians = math.radians(angle)
            marker_x = center_x + radius * math.sin(roll_angle_radians)
            marker_y = center_y - radius * math.cos(roll_angle_radians)
            self.canvas.create_text(marker_x, marker_y, text=f"{angle:+}", fill="#ffffff", font=("Helvetica", 10, "bold"))

        # Draw yaw indicator
        yaw_radius = radius + 20
        yaw_angle_radians = math.radians(yaw)
        yaw_x = center_x + yaw_radius * math.sin(yaw_angle_radians)
        yaw_y = center_y - yaw_radius * math.cos(yaw_angle_radians)
        self.canvas.create_text(yaw_x, yaw_y, text="YAW", fill="#ff0000", font=("Helvetica", 12, "bold"))

        # Display pitch, roll, and yaw values
        horizon_text = "Pitch: {:+6.2f}°   Roll: {:+6.2f}°   Yaw: {:+6.2f}°".format(pitch, roll, yaw)
        self.canvas.create_text(center_x, y0 + height + 20, text=horizon_text, fill="#ffffff", font=("Helvetica", 16, "bold"))

    def draw_fuel_meter(self):
        x0, y0, width, height = 50, 300, 500, 30
        self.canvas.create_rectangle(x0, y0, x0 + width, y0 + height, outline="#ffffff", width=2)
        fuel_width = (self.fuel_level / 100.0) * width
        self.canvas.create_rectangle(x0, y0, x0 + fuel_width, y0 + height, fill="#00ff00", outline="")
        fuel_text = "Fuel Level: {:.2f}%".format(self.fuel_level)
        self.canvas.create_text(x0 + width / 2, y0 + height / 2, text=fuel_text, fill="#ffffff", font=("Helvetica", 14, "bold"))

    def draw_thruster_and_legs(self):
        center_x, center_y = 800, 200
        main_radius = 30

        self.canvas.create_oval(center_x - main_radius, center_y - main_radius,
                                center_x + main_radius, center_y + main_radius,
                                fill="#444444", outline="#ffffff", width=4)
        main_text = "Main\n{:.1f}%".format(self.throttles.get(7, 0.0) * 100)
        self.canvas.create_text(center_x, center_y, text=main_text, fill="#ffffff", font=("Helvetica", 12, "bold"))

        offset = 100
        thruster_radius = 25
        pos8 = (center_x, center_y - offset)
        self.draw_thruster_circle(pos8, thruster_radius, self.throttles.get(8, 0.0), "T4")
        pos10 = (center_x, center_y + offset)
        self.draw_thruster_circle(pos10, thruster_radius, self.throttles.get(10, 0.0), "T2")
        pos9 = (center_x - offset, center_y)
        self.draw_thruster_circle(pos9, thruster_radius, self.throttles.get(9, 0.0), "T3")
        pos11 = (center_x + offset, center_y)
        self.draw_thruster_circle(pos11, thruster_radius, self.throttles.get(11, 0.0), "T1")

        hex_radius = 150
        hex_points = []
        for i in range(6):
            angle = math.radians(60 * i)
            hx = center_x + hex_radius * math.cos(angle)
            hy = center_y + hex_radius * math.sin(angle)
            hex_points.append((hx, hy))
        flat_hex_points = [coord for point in hex_points for coord in point]
        self.canvas.create_polygon(flat_hex_points, outline="#ffaa00", fill="", width=2)

        leg_radius = 15
        leg_positions = []
        for i in range(6):
            angle = math.radians(60 * i + 90)
            r = 150
            lx = center_x + r * math.cos(angle)
            ly = center_y + r * math.sin(angle)
            leg_positions.append((lx, ly))
        for idx, (lx, ly) in enumerate(leg_positions):
            color = "#00ff00" if self.leg_contacts[idx + 1] else "#ff0000"
            self.canvas.create_oval(lx - leg_radius, ly - leg_radius,
                                     lx + leg_radius, ly + leg_radius,
                                     fill=color, outline="#ffffff", width=2)
            self.canvas.create_text(lx, ly, text="L{}".format(idx + 1), fill="#ffffff", font=("Helvetica", 10, "bold"))

    def draw_thruster_circle(self, pos, radius, throttle, label):
        x, y = pos
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                fill="#003366", outline="#ffffff", width=2)
        text = "{}\n{:.1f}%".format(label, throttle * 100)
        self.canvas.create_text(x, y, text=text, fill="#ffffff", font=("Helvetica", 10, "bold"))

    # New method: Display additional telemetry information (if not already added)
    def draw_additional_info(self):
        x0, y0 = 50, 620
        info_text = (
            f"Vertical Velocity: {self.vertical_velocity:6.2f} m/s    "
            f"Horizontal Speed: {self.horizontal_speed:6.2f} m/s\n"
            f"Time-to-Go: {self.time_to_go:6.2f} s    "
            f"Engine Temp: {self.engine_temperature:6.2f} °C"
        )
        self.canvas.create_text(x0, y0, anchor="nw", text=info_text, fill="#ffffff", font=("Helvetica", 14, "bold"))

    def draw_telemetry(self):
        self.canvas.delete("all")
        self.draw_artificial_horizon()
        self.draw_fuel_meter()
        self.draw_thruster_and_legs()
        self.draw_altitude_graph()
        self.draw_additional_info()
        self.draw_topdown_trajectory()
        # self.time_step += 1

    def update_telemetry(self):
        self.draw_telemetry()
        self.root.update_idletasks()

    def simulate(self):
        i = self.time_step
        self.imu_state["pitch"] = 15 * math.sin(i / 10.0)
        self.imu_state["roll"] = 15 * math.cos(i / 10.0)
        self.fuel_level = max(0, self.fuel_level - 0.1)
        self.throttles[7] = 0.5 + 0.5 * math.sin(i / 15.0)
        self.throttles[8] = 0.5 + 0.5 * math.sin((i + 2) / 15.0)
        self.throttles[9] = 0.5 + 0.5 * math.sin((i + 4) / 15.0)
        self.throttles[10] = 0.5 + 0.5 * math.sin((i + 6) / 15.0)
        self.throttles[11] = 0.5 + 0.5 * math.sin((i + 8) / 15.0)

        self.lidar = [50 + 10 * math.sin((i + j) / 20.0) for j in range(6)]

        if i > 150:
            self.leg_contacts = [True, True, False, False, True, True]
        else:
            self.leg_contacts = [False, False, False, False, False, False]

        # Simulate additional telemetry values
        self.vertical_velocity = -abs(5 * math.sin(i / 20.0))  # m/s, descending
        self.horizontal_speed = abs(3 * math.cos(i / 25.0))      # m/s
        valid_lidar = [v for v in self.lidar if v > 0]
        altitude = sum(valid_lidar) / len(valid_lidar) if valid_lidar else 0
        self.time_to_go = altitude / (abs(self.vertical_velocity) + 0.1)
        self.engine_temperature = 25 + (100 - self.fuel_level) * 0.3

        # Update horizontal position and trajectory history
        self.position_xy = (10 * math.sin(i / 30.0), 10 * math.cos(i / 30.0))
        self.position_history.append(self.position_xy)
        if len(self.position_history) > 200:
            self.position_history.pop(0)

        self.draw_telemetry()
        self.draw_additional_info()
        self.draw_topdown_trajectory()
        self.time_step += 1
        self.root.after(100, self.update_telemetry)

if __name__ == "__main__":
    root = tk.Tk()
    display = TelemetryDisplay(root)
    root.mainloop()
