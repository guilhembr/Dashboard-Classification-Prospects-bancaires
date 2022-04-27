web: gunicorn -w 4 -k uvicorn.workers.UvicornWorker --pythonpath P7_SCORING api:app --host=0.0.0.0 --port=${PORT:-5000}

