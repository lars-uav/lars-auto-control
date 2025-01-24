from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
import json
import numpy as np
import asyncio
from typing import Set

# Database Models
Base = declarative_base()

class DroneDataDB(Base):
    __tablename__ = "drone_data"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime)
    ndvi_mean = Column(Float)
    ndvi_max = Column(Float)
    ndvi_min = Column(Float)
    inference_data = Column(String)
    lat = Column(Float, nullable=True)
    lon = Column(Float, nullable=True)
    image_path = Column(String, nullable=True)

# Pydantic Models
class DroneDataIn(BaseModel):
    timestamp: float
    ndvi_mean: float
    ndvi_max: float
    ndvi_min: float
    inference_results: List[float]
    gps_coords: Optional[tuple]
    image: Optional[str]

class DroneDataOut(BaseModel):
    id: int
    timestamp: datetime
    ndvi_mean: float
    ndvi_max: float
    ndvi_min: float
    inference_results: List[float]
    gps_coords: Optional[tuple]

# Database Connection
SQLALCHEMY_DATABASE_URL = "sqlite:///./drone_data.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

# FastAPI App
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connections store
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, data: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except:
                await self.disconnect(connection)

manager = ConnectionManager()

# Routes
@app.post("/data", response_model=DroneDataOut)
async def receive_data(data: DroneDataIn):
    try:
        db = SessionLocal()
        db_data = DroneDataDB(
            timestamp=datetime.fromtimestamp(data.timestamp),
            ndvi_mean=data.ndvi_mean,
            ndvi_max=data.ndvi_max,
            ndvi_min=data.ndvi_min,
            inference_data=json.dumps(data.inference_results),
            lat=data.gps_coords[0] if data.gps_coords else None,
            lon=data.gps_coords[1] if data.gps_coords else None
        )
        
        db.add(db_data)
        db.commit()
        db.refresh(db_data)
        
        # Broadcast to WebSocket clients
        await manager.broadcast({
            "id": db_data.id,
            "timestamp": data.timestamp,
            "ndvi_mean": data.ndvi_mean,
            "ndvi_max": data.ndvi_max,
            "ndvi_min": data.ndvi_min,
            "inference_results": data.inference_results,
            "gps_coords": data.gps_coords
        })
        
        return db_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/data/recent/{limit}", response_model=List[DroneDataOut])
async def get_recent_data(limit: int = 100):
    db = SessionLocal()
    try:
        data = db.query(DroneDataDB)\
            .order_by(DroneDataDB.timestamp.desc())\
            .limit(limit)\
            .all()
        return data
    finally:
        db.close()

@app.get("/data/stats")
async def get_stats():
    db = SessionLocal()
    try:
        data = db.query(DroneDataDB).all()
        if not data:
            return {
                "ndvi_mean": 0,
                "ndvi_max": 0,
                "ndvi_min": 0,
                "total_records": 0
            }
            
        ndvi_means = [d.ndvi_mean for d in data]
        return {
            "ndvi_mean": np.mean(ndvi_means),
            "ndvi_max": max(d.ndvi_max for d in data),
            "ndvi_min": min(d.ndvi_min for d in data),
            "total_records": len(data)
        }
    finally:
        db.close()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle any incoming WebSocket messages if needed
            await asyncio.sleep(0.1)
    except:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)