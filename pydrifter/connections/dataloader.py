import os
from abc import ABC
import dataclasses
from typing import Union
from pydrifter.connections.s3 import S3Loader, S3Config
import boto3


@dataclasses.dataclass
class DataLoader(ABC):
    s3_config: Union[S3Config, None] = None
    postgres_config: Union[str, None] = None
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

    def s3_configuration(self):
        return self.s3_config

    def read_from_s3(self, bucket_name, file_path, *args, **kwargs):
        if not self.s3_config:
            raise ValueError("Define S3Config first")

        return S3Loader.read(
            self.s3_connection,
            bucket_name,
            file_path,
            *args, **kwargs
        )

    def save_to_s3(self, bucket_name, file_path, file, *args, **kwargs):
        if not self.s3_config:
            raise ValueError("Define S3Config first")

        return S3Loader.save(
            self.s3_connection, bucket_name, file_path, file, *args, **kwargs
        )
