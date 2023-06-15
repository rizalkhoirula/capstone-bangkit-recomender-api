from app import Models
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
model = Models()

@app.get('/')
async def root():
    return {'status': 'success'}


@app.get("/recomendation")
async def predict(id: int = Query(..., description="rekomendasi untuk user id berapa?")):
    return {"status": "success",
            "result": model.predict_to_json(user_id=id, ascending=False)
            }

# Menjalankan server dengan uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)

"""
running using terminal:
python -B -m uvicorn main:app --reload
"""


