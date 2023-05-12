from fastapi import FastAPI
import uvicorn
from model import InvkineModel, Pose, Thetas
import numpy as np

app = FastAPI()
model = InvkineModel(model_path = 'model.h5')

@app.on_event('startup')
def load_model():
    model.load()

@app.get("/")
def home():
    return {"Hello": "World"}

@app.post('/inference')
def inference(pose: Pose)->Thetas:
    prediction = model.predict(pose)
    thetas = Thetas(theta0=prediction[0], theta1=prediction[1])
    return thetas

if __name__ == '__main__':
    uvicorn.run(app, port=8000)