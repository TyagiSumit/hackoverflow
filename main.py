import uvicorn
from fastapi import FastAPI
import ml
app = FastAPI()

@app.get('/')
async def index():
    return {"text" : "Hello Masters"}

@app.get('/location/{location}')
async def get_location(location):
    lat, long = [float(x) for x in location.split(",")]
    return str(ml.shortest_entry(lat, long))

@app.get('/disease/{symptoms}')
async def predict_disease(symptoms):
    return str(ml.predict(symptoms))



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

