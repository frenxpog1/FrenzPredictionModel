from fastapi import FastAPI, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from database import SessionLocal, init_db
import models

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
def startup_event():
    init_db()

@app.get("/")
@app.get("/index.html")
def read_root(request: Request, db: Session = Depends(get_db)):
    matches = db.query(models.Match).order_by(models.Match.match_timestamp.desc()).all()
    patches = db.query(models.Patch).all()
    heroes = db.query(models.Hero).all()
    items = db.query(models.Item).all()
    return templates.TemplateResponse(
        request,
        "index.html", 
        {
            "matches": matches,
            "patches": patches,
            "heroes": heroes,
            "items": items
        }
    )

@app.get("/heroes")
@app.get("/heroes.html")
def list_heroes(request: Request, db: Session = Depends(get_db)):
    heroes = db.query(models.Hero).all()
    return templates.TemplateResponse(request, "heroes.html", {"heroes": heroes})

@app.get("/items")
@app.get("/items.html")
def list_items(request: Request, db: Session = Depends(get_db)):
    items = db.query(models.Item).all()
    return templates.TemplateResponse(request, "items.html", {"items": items})
