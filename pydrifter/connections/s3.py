from abc import ABC
import dataclasses
import pandas as pd
from PIL import Image
import io
import json
from omegaconf import OmegaConf

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
    def table_extensions():
        return {
            "csv": pd.read_csv,
            "xlsx": pd.read_excel,
            "parquet": pd.read_parquet
        }

    @staticmethod
    def image_extensions():
        return {"jpg", "jpeg", "png"}

    @staticmethod
    def yaml_extensions():
        return {"yml", "yaml"}

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

        if file_extension in S3Loader.table_extensions():
            logger.info(f"[TABLE] Downloaded from 's3://{bucket_name}/{file_path}'. Size: {size_mb:.2f} MB")
            return S3Loader.table_extensions()[file_extension](buffer)
        elif file_extension in S3Loader.image_extensions():
            logger.info(f"[IMAGE] Downloaded from 's3://{bucket_name}/{file_path}'. Size: {size_mb:.2f} MB")
            return Image.open(buffer)
        elif file_extension in S3Loader.yaml_extensions():
            logger.info(f"[CONFIG] Downloaded from 's3://{bucket_name}/{file_path}'. Size: {size_mb:.2f} MB")
            return OmegaConf.load(buffer)
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
        else:
            raise TypeError("Unknown datatype of the file")

        buffer.seek(0)
        size_mb = len(buffer.getvalue()) / (1024 * 1024)
        try:
            s3_connection.upload_fileobj(buffer, f"{bucket_name}", f"{file_path}")
            logger.info(f"Successfully uploaded to 's3://{bucket_name}/{file_path}'. File size: {size_mb:.2f} MB")
        except Exception as e:
            raise e

    @staticmethod
    def delete(s3_connection, bucket_name: str, file_path: str):
        try:
            s3_connection.delete_object(
                Bucket=f"{bucket_name}",
                Key=f"{file_path}"
            )
            logger.info(f"Successfully removed 's3://{bucket_name}/{file_path}'")
        except Exception as e:
            raise e

    def show(s3_connection, bucket_name: str):
        total_files = len(s3_connection.list_objects(Bucket=bucket_name)["Contents"])
        print(f"Total files in '{bucket_name}': {total_files:,}\n")

        return list(s3_connection.list_objects(Bucket=bucket_name)["Contents"])
        # for key in s3_connection.list_objects(Bucket=bucket_name)["Contents"]:
        #     print(key['Key'])
