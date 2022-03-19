from pyrsistent import v
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional



app = FastAPI()


class BookIdGenerator:
    def __init__(self):
        self.current_id = 1
    
    def next_id(self):
        self.current_id += 1
        return self.current_id

class Book(BaseModel):
    id: int
    title: str
    author: Optional[str] = None

class CreateBookRequest(BaseModel):
    title: str
    author: Optional[str] = None

class UpdateBookRequest(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None

books = [
    Book(id=1, title='Harry Potter', author='J K Rowling')
]

book_id_gen = BookIdGenerator()

@app.get('/books')
def get_all_books():
    return books

@app.get('/books/{book_id}')
def get_book(book_id: int):
    for book in books:
        if book.id == book_id:
            return {
                'book': book
            }

@app.post("/books")
async def create_book(cbr: CreateBookRequest):
    book = Book(
        id=book_id_gen.next_id(),
        title=cbr.title,
        author=cbr.author
    )

    books.append(book)

    return book

@app.patch('/books/{book_id}')
def update_book(book_id: int, ubr: UpdateBookRequest):
    for book in books:
        if book.id == book_id:
            if not ubr.title is None:
                book.title = ubr.title
            if not ubr.author is None:
                book.author = ubr.author
            return book

if __name__ == '__main__':
    uvicorn.run("main:app")
