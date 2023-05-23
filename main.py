import uvicorn
from fastapi import FastAPI
from models.model import InvkineModel, Pose, Thetas

app = FastAPI()
model = InvkineModel(model_path = 'data/model.h5')

@app.on_event('startup')
def load_model():
    model.load()

@app.get("/")
def home():
    return {"Hello": "World"}

@app.post('/inference')
def inference(pose: Pose)->Thetas:
    prediction = model.predict(pose)
    thetas = Thetas(theta0=prediction[0], theta1=prediction[1], theta2=prediction[2])
    return thetas

if __name__ == '__main__':
    uvicorn.run(app, port=8000)