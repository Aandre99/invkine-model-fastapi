from pydantic import BaseModel
from pathlib import Path
import tensorflow as tf
import numpy as np 
import time

class Pose(BaseModel):
    
    x: float
    y: float
    z: float
    

class Thetas(BaseModel):
    
    theta0: float
    theta1: float
    theta2: float

    time: float
    
class InvkineModel:
    
    def __init__(self, model_path:str) -> None:
        self.model_path = Path(model_path)
        self.model = None
        
    def load(self) -> None:
        self.model = tf.keras.models.load_model(self.model_path)
        
    def predict(self, x:Pose) -> np.ndarray:
        pose_array = np.array([[x.x, x.y, x.z]])
        print(pose_array)
        begin_time = time.time()
        prediction = self.model.predict(pose_array)[0]
        end_time = time.time() - begin_time
        return end_time, prediction