"""Notebook access policy; no GPU, tunnel, credentials or model download needed."""
import os
import secrets
import sys
import tempfile

sys.path.insert(0, '.')
with tempfile.TemporaryDirectory() as data:
    url = 'https://notebook.example'
    os.environ.update(CLIPKONTRAST_DATA=data, CLIPKONTRAST_PUBLIC_SIGNUP='0', CLIPKONTRAST_ACCOUNTS='1',
                      CLIPKONTRAST_PUBLIC_URL=url, DATABASE_URL='', GOOGLE_CLIENT_ID='test', GOOGLE_CLIENT_SECRET='test')
    from fastapi.testclient import TestClient
    from deploy.kaggle.launch import OWNER, provision
    from studio import auth, server

    password = secrets.token_urlsafe(24)
    provision(password)
    provision(password)  # restarting must not create another account or change its password
    client = TestClient(server.app, base_url=url, follow_redirects=False)
    assert client.get('/app').headers['location'] == '/login'
    assert client.get('/api/jobs').status_code == 401
    assert client.get('/auth/google/start').status_code == 404
    assert client.post('/api/auth/signup', json={}).status_code == 403
    assert client.post('/api/auth/login', json={'email': OWNER, 'password': 'incorrect'}).status_code == 400
    r = client.post('/api/auth/login', json={'email': OWNER, 'password': password})
    assert r.status_code == 200 and 'secure' in r.headers['set-cookie'].lower()
    assert client.get('/api/auth/me').json()['user']['is_admin']
    assert client.get('/app').status_code == 200
    assert client.put('/api/settings', json={}, headers={'Sec-Fetch-Site': 'cross-site'}).status_code == 403
    assert client.post('/api/storage/open').status_code == 400
    assert len(auth.user_ids()) == 1
    print('PASS private notebook access, signup policy, restart and owner login')
