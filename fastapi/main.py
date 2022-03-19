import uvicorn

from fastapi import FastAPI

app = FastAPI()

books = [
    {
        'id': 1,
        'title': 'Harry Potter',
        'author': 'J K Rowling'
    }
]

@app.get('/books/{book_id}')
def get_book(book_id: int):
    for book in books:
        if book['id'] == book_id:
            return {
                'book': book
            }

if __name__ == '__main__':
    uvicorn.run("main:app")
