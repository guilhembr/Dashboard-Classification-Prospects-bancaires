web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app --host=0.0.0.0 --port=${PORT:-5000}

