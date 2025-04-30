from abc import ABC
import dataclasses
import io
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import psycopg2

from pydrifter.logger import create_logger

logger = create_logger(level="info")

@dataclasses.dataclass
class PostgresConfig(ABC):
    username: str
    password: str
    host: str
    port: str
    database: str

    def __repr__(self):
        return f"PostgresConfig(username='{self.username[0]}***', password='{self.password[0]}***', host='{self.host}:{self.port}/{self.database}')"

    def connection_engine(self) -> Engine:
        return create_engine(
            url=f"postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        )

    def database_params(self):
        return {
            "user": self.username,
            "password": self.password,
            "database": self.database,
            "host": self.host,
            "port": self.port,
        }


@dataclasses.dataclass
class PostgresLoader(ABC):

    @staticmethod
    def read(postgres_connection: PostgresConfig, sql: str):
        return pd.read_sql(
            sql=sql,
            con=postgres_connection
        )

    @staticmethod
    def save(
        postgres_connection: PostgresConfig,
        schema: str,
        table_name: str,
        data: pd.DataFrame,
        chunk_size: int | None = None,
        if_exists: str = "append"
    ):
        return data.to_sql(
            name=table_name,
            con=postgres_connection.connection_engine(),
            schema=schema,
            index=False,
            if_exists=if_exists,
            chunksize=chunk_size,
        )

        # TODO: figure out with this garbage
        
        # for col in data.columns:
        #     if pd.api.types.is_datetime64_any_dtype(data[col]):
        #         data[col] = data[col].dt.strftime("%Y-%m-%d")

        # columns = data.columns.tolist()
        # columns_str = ', '.join(columns)
        # columns_str = ", ".join([f'"{col}"' for col in columns])
        # sss = " ".join(['%s,' for _ in range(len(columns))])[:-1]

        # sql = f"INSERT INTO {schema}.{table_name} ({columns_str}) VALUES ({sss});"
        # data_tuples = [tuple(row) for row in data.itertuples(index=False, name=None)]

        # connection = psycopg2.connect(**postgres_connection.database_params())
        # cursor = connection.cursor()

        # # with connection.cursor() as cursor:
        # #     cursor.executemany(sql, data_tuples, commit=False)
        # # connection.commit()
        # # connection.close()

        # try:
        #     for i in range(0, len(data), chunk_size):
        #         chunk = data.iloc[i:i + chunk_size]
        #         data_tuples = [tuple(row) for row in chunk.itertuples(index=False, name=None)]
        #         cursor.executemany(sql, data_tuples)
        #         connection.commit()
        #         print(i)
        # except Exception as e:
        #     connection.rollback()
        #     raise e
        # finally:
        #     cursor.close()
        #     connection.close()
