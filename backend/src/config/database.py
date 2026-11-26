from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .settings import DATABASE_URL


class Base(DeclarativeBase):
    pass


engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Session = sessionmaker(bind=engine, expire_on_commit=False)


@event.listens_for(engine, 'connect')
def configure_sqlite(connection, _):
    if engine.dialect.name == 'sqlite':
        connection.execute('PRAGMA foreign_keys = ON')


def session_scope():
    with Session() as session:
        yield session
