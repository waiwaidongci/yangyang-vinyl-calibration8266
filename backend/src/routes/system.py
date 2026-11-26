from fastapi import APIRouter
from sqlalchemy import text

from src.controllers.dependencies import Db
from src.domain.project import SPEC

router = APIRouter()


@router.get('/health')
def health(session: Db):
    session.execute(text('SELECT 1'))
    return {'status': 'ok'}


@router.get('/spec')
def project_spec():
    return SPEC
