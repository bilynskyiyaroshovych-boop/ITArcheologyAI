from starlette.testclient import TestClient
from api import app

client = TestClient(app)
print('GET / ->', client.get('/').status_code)
print('GET /health ->', client.get('/health').status_code, client.get('/health').json())
