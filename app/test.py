from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Todo(BaseModel):
    id: int
    title: str


@app.get('/')
def homes():
    return {'messages': 'Hello World'}

@app.get('/')
def get_todos():
    return [
        {'id': 1, 'title': 'Learn APIs'},
        {'id': 2, 'title': 'Building FastAPI project'},
        {'id': 3, 'title': 'Learn MLOps'},
        {'id': 4, 'title': 'Deploy Elvara Sepsis Model'}
    ]

@app.post('/todos')
def create_todo(todo: Todo):
    return todo