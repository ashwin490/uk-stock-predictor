"""Create local Streamlit login secrets without a hardcoded login password."""
import getpass
import hashlib
import os
from pathlib import Path

secrets = Path(__file__).resolve().parent / '.streamlit' / 'secrets.toml'
if secrets.exists():
    raise SystemExit(f'{secrets} already exists. For safety this helper will not overwrite it.')
username = input('Choose a login username [admin]: ').strip() or 'admin'
if '\n' in username or '"' in username or len(username) > 100:
    raise SystemExit('Unsupported username characters')
p1 = getpass.getpass('Choose a new unique password (16+ characters): ')
p2 = getpass.getpass('Repeat the password: ')
if p1 != p2 or len(p1) < 16:
    raise SystemExit('Passwords did not match or were too short (minimum 16 characters).')
secrets.parent.mkdir(parents=True, exist_ok=True)
with secrets.open('x', encoding='utf-8') as f:
    f.write(f'AUTH_USER = "{username}"\nAUTH_PASS_HASH = "{hashlib.sha256(p1.encode()).hexdigest()}"\n')
try:
    os.chmod(secrets, 0o600)
except OSError:
    pass
print('Local credentials saved in .streamlit/secrets.toml. Do NOT upload this file to GitHub.')
