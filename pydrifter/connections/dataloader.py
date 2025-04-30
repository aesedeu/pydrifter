import os
import pandas as pd
from abc import ABC
import dataclasses
from typing import Union
import boto3

from pydrifter.connections.s3 import S3Loader, S3Config
from pydrifter.connections.postgres import PostgresLoader, PostgresConfig


@dataclasses.dataclass
class DataLoader(ABC):
    s3_config: Union[S3Config, None] = None
    postgres_config: Union[PostgresConfig, None] = None
    oracle_config: Union[str, None] = None

    def __post_init__(self):
        if self.s3_config is not None and not isinstance(self.s3_config, S3Config):
            raise TypeError(f"s3_config must be an instance of S3Config, got {type(self.s3_config).__name__}")
        elif self.s3_config is not None and isinstance(self.s3_config, S3Config):
            self.__s3_session = boto3.session.Session(
                aws_access_key_id=self.s3_config.access_key,
                aws_secret_access_key=self.s3_config.secret_key,
            )
            self.s3_connection = self.__s3_session.client(
                service_name="s3", endpoint_url=self.s3_config.url
            )

    def s3_info(self):
        return self.s3_config

    def postgres_info(self):
        return self.postgres_config

    def oracle_info(self):
        return self.oracle_config

    ### S3 ###
    def read_from_s3(self, bucket_name, file_path, *args, **kwargs):
        if not self.s3_config:
            raise ValueError("Define S3Config first")

        return S3Loader.read(
            self.s3_connection, bucket_name, file_path, *args, **kwargs
        )

    def save_to_s3(self, bucket_name, file_path, file, *args, **kwargs):
        if not self.s3_config:
            raise ValueError("Define S3Config first")

        return S3Loader.save(
            self.s3_connection, bucket_name, file_path, file, *args, **kwargs
        )

    def delete_from_s3(self, bucket_name, file_path, *args, **kwargs):
        if not self.s3_config:
            raise ValueError("Define S3Config first")

        return S3Loader.delete(
            self.s3_connection, bucket_name, file_path, *args, **kwargs
        )

    def show_s3(self, bucket_name):
        return S3Loader.show(
            self.s3_connection, bucket_name
        )

    ### POSTGRES ###
    def read_from_postgres(self, sql, *args, **kwargs):
        if not self.postgres_config:
            raise ValueError("Define PostgresConfig first")

        return PostgresLoader.read(
            postgres_connection=self.postgres_config.connection_engine(),
            sql=sql
        )

    def save_to_postgres(
        self,
        schema: str,
        table_name: str,
        data: pd.DataFrame,
        chunk_size: int | None = None,
        if_exists: str = "append"
    ):
        if not self.postgres_config:
            raise ValueError("Define PostgresConfig first")

        return PostgresLoader.save(
            postgres_connection=self.postgres_config,
            schema=schema,
            table_name=table_name,
            data=data,
            chunk_size=chunk_size,
            if_exists=if_exists,
        )
