
## Run FLASK APP

### Run sync using WSGI

gunicorn app:app --workers 4 --bind 0.0.0.0:8080


## RUN FASTAPI APP

### Run async using ASGI

gunicorn app:app -k uvicorn.workers.UvicornWorker --workers 4 --bind 0.0.0.0:8080
