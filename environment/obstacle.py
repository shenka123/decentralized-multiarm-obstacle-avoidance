import pybullet as p
import numpy as np

class Obstacle:
    
    def __init__(self, config, base_position):

        self.type = config['type']
        self.config = config
        self.color =  [0.5, 0.8, 1.0, 0.9]
        self.mass = 0
        self.base_position = base_position
        
        if self.type == "Sphere":
            self.radius = config['radius']
            self.construct_sphere()


    def construct_sphere(self):

        # Create visual shape (what you see)
        self.visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.radius,
            rgbaColor=self.color
        )
        
        # Create collision shape (for collision detection)
        self.collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=self.radius
        )
        
        # Create the multi-body (combines visual + collision)
        self.body_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=self.collision_shape_id,
            baseVisualShapeIndex=self.visual_shape_id,
            basePosition=self.base_position,
            baseOrientation=[0, 0, 0, 1]
        )
    
    # def check_collision_with_body(self, body_id, collision_distance=0.0):
    #     """
    #     Check if this sphere collides with a specific body.
        
    #     Args:
    #         body_id: PyBullet body ID to check collision with
    #         collision_distance: distance threshold for collision (meters)
            
    #     Returns:
    #         bool: True if collision detected
    #     """
    #     contact_points = p.getClosestPoints(
    #         bodyA=self.body_id,
    #         bodyB=body_id,
    #         distance=collision_distance + self.radius
    #     )
        
    #     for point in contact_points:
    #         if point[8] <= collision_distance:  # point[8] is distance
    #             return True
        
    #     return False
    
    # def check_collision_with_ur5(self, ur5, collision_distance=0.0):
    #     """
    #     Check if this sphere collides with a UR5 robot.
        
    #     Args:
    #         ur5: UR5 robot object with body_id attribute
    #         collision_distance: distance threshold for collision
            
    #     Returns:
    #         bool: True if collision detected
    #     """
    #     return self.check_collision_with_body(ur5.body_id, collision_distance)
    
    # def check_collision_with_ur5s(self, ur5s, collision_distance=0.0):
    #     """
    #     Check if this sphere collides with any robot in a list.
        
    #     Args:
    #         ur5s: List of UR5 robot objects
    #         collision_distance: distance threshold for collision
            
    #     Returns:
    #         tuple: (collision_detected, ur5_index)
    #             collision_detected: bool
    #             ur5_index: int index of first colliding robot, or None
    #     """
    #     for idx, ur5 in enumerate(ur5s):
    #         if self.check_collision_with_ur5(ur5, collision_distance):
    #             return True, idx
    #     return False, None
    
    def highlight_collision(self):
        """Change sphere color to bright red to indicate collision."""
        collision_color = [1.0, 0.0, 0.0, 1.0]
        p.changeVisualShape(self.body_id, -1, rgbaColor=collision_color)
    
    def reset_color(self):
        """Reset sphere to original color."""
        p.changeVisualShape(self.body_id, -1, rgbaColor=self.original_color)

    def disable(self, idx=0):
        self.enabled = False
        self.set_position([2*idx, 20, 10])
        # self.reset()
        # self.step()

    def enable(self):
        self.enabled = True
    

    def set_position(self, position):
        """Move sphere to new position."""
        self.position = list(position)
        p.resetBasePositionAndOrientation(
            self.body_id,
            self.position,
            [0, 0, 0, 1]
        )

    def get_position(self):
        """Get current sphere position."""
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        return list(pos)
    
    def remove(self):
        """Remove sphere from simulation."""
        if self.body_id is not None:
            p.removeBody(self.body_id)
            self.body_id = None
