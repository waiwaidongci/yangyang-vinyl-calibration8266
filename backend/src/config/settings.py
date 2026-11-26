import os

DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///local.db')
JWT_SECRET = os.getenv('JWT_SECRET', 'only-for-local-development-please-change')
SEED_DEMO = os.getenv('SEED_DEMO', 'true').lower() == 'true'
