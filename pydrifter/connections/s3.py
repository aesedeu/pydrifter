from abc import ABC
import dataclasses
from typing import Any
import pandas as pd
from PIL import Image
import io
from boto3.exceptions import S3UploadFailedError

from pydrifter.logger import create_logger

logger = create_logger(level="info")

@dataclasses.dataclass
class S3Config(ABC):
    access_key: str
    secret_key: str
    url: str

    def __repr__(self):
        return f"S3Config(access_key='{self.access_key[0]}***', secret_key='{self.secret_key[0]}***', url='{self.url}')"


@dataclasses.dataclass
class S3Loader(ABC):

    @staticmethod
    def table_extentions():
        return {
            "csv": pd.read_csv,
            "xlsx": pd.read_excel,
            "parquet": pd.read_parquet
        }

    @staticmethod
    def image_extensions():
        return {"jpg", "jpeg", "png"}

    @staticmethod
    def read(s3_connection, bucket_name, file_path: str):
        obj = s3_connection.get_object(
            Bucket=f"{bucket_name}",
            Key=f"{file_path}",
        )
        raw_data = obj["Body"].read()
        buffer = io.BytesIO(raw_data)
        size_mb = len(buffer.getvalue()) / (1024 * 1024)

        file_extension = file_path.split(".")[-1]

        if file_extension in S3Loader.table_downloaders():
            logger.info(f"[TABLE] Downloaded from 's3://{bucket_name}/{file_path}'. Size: {size_mb:.2f} MB")
            return S3Loader.table_downloaders()[file_extension](buffer)
        elif file_extension in S3Loader.image_extensions():
            logger.info(f"[IMAGE] Downloaded from 's3://{bucket_name}/{file_path}'. Size: {size_mb:.2f} MB")
            return Image.open(buffer)
        else:
            raise TypeError(f"Unsupported file extension '{file_extension}'")

    @staticmethod
    def save(s3_connection, bucket_name: str, file_path: str, file):
        buffer = io.BytesIO()
        file_extension = file_path.split(".")[-1]

        if file_extension == "parquet":
            file.to_parquet(buffer, index=False, engine="pyarrow")
        elif file_extension == "csv":
            file.to_csv(buffer, index=False)
        elif file_extension in S3Loader.image_extensions():
            if isinstance(file, Image.Image):
                file.save(buffer, format=file_extension.upper())
            else:
                raise TypeError("Expected a PIL.Image.Image object for image upload.")

        buffer.seek(0)
        size_mb = len(buffer.getvalue()) / (1024 * 1024)
        try:
            s3_connection.upload_fileobj(buffer, f"{bucket_name}", f"{file_path}")
            logger.info(f"Successfully uploaded to 's3://{bucket_name}/{file_path}'. File size: {size_mb:.2f} MB")
        except Exception as e:
            raise e
