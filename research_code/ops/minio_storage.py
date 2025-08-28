from minio import Minio
from minio.error import S3Error
import os
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
def ensure_bucket_exists(client, bucket_name):
    if not client.bucket_exists(bucket_name):
        print(f"Bucket '{bucket_name}' does not exist. Creating bucket.")
        client.make_bucket(bucket_name)
    else:
        print(f"Bucket '{bucket_name}' already exists.")


def upload_folder(client, bucket_name, local_folder, remote_folder, allowed_extensions=None):
    for root, dirs, files in os.walk(local_folder):
        # Skip hidden folders
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        files = [f for f in files if not f.startswith('.')]

        for file_name in tqdm(files, total=len(files), desc=f"Uploading to {remote_folder}"):
            if allowed_extensions and not file_name.lower().endswith(tuple(allowed_extensions)):
                continue

            local_file_path = os.path.join(root, file_name)
            remote_object_name = f"{remote_folder.rstrip('/')}/{file_name}"

            try:
                # Check if object already exists in MinIO
                client.stat_object(bucket_name, remote_object_name)
                # print(f"⏭️  Skipped (already exists): {remote_object_name}")
                continue
            except S3Error as e:
                if e.code == "NoSuchKey":
                    # Object doesn't exist, so upload it
                    client.fput_object(bucket_name, remote_object_name, local_file_path)
                    print(f"Uploaded to {remote_object_name}")
                    continue
def download_from_minio(
    client: Minio,
    bucket_name: str,
    remote_prefix: str,
    local_target_dir: str
):
    """
    Downloads all objects from a given MinIO prefix to a local directory.

    :param client: Minio client
    :param bucket_name: Name of the MinIO bucket
    :param remote_prefix: Remote folder/prefix in the bucket
    :param local_target_dir: Local folder to save files
    """
    local_target_dir = os.path.abspath(local_target_dir)
    os.makedirs(local_target_dir, exist_ok=True)

    objects = client.list_objects(bucket_name, prefix=remote_prefix, recursive=True)
    for obj in tqdm(objects, desc=f"Downloading from '{remote_prefix}'"):
        file_name = os.path.basename(obj.object_name)

        # Skip hidden files
        if file_name.startswith('.'):
            continue

        # Determine local file path
        relative_path = os.path.relpath(obj.object_name, start=remote_prefix)
        local_file_path = os.path.join(local_target_dir, relative_path)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Skip if file already exists
        if os.path.exists(local_file_path):
            print(f"⏭️  Skipped (already exists): {relative_path}")
            continue

        try:
            client.fget_object(bucket_name, obj.object_name, local_file_path)
            print(f"✅ Downloaded: {relative_path}")
        except S3Error as err:
            print(f"❌ Failed to download {obj.object_name}: {err}")

def main_back_up_images(cfg: DictConfig):
    bucket_name = cfg.minio.bucket_name
    remote_folder = cfg.minio.folder_name
    local_folder =f"{cfg.storage.sds}/{cfg.datasets.svi_dir}"

    minio_id = cfg.minio.silverways_account
    minio_key = cfg.minio.silverways_key

    allowed_extensions = ['.jpg', '.jpeg', '.png', '.tiff']  # Specify desired file formats here
    client = Minio(
        endpoint = cfg.minio.endpoint,#"hot.storage.heigit.org",
        access_key=minio_id,
        secret_key=minio_key,
        secure=True
    )
    try:
        ensure_bucket_exists(client, bucket_name)
        upload_folder(client, bucket_name, local_folder, remote_folder, allowed_extensions)
    except S3Error as err:
        print("S3 Error:", err)
    except Exception as err:
        print("General Error:", err)
@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def hydra_main(cfg: DictConfig):
    return main_back_up_images(cfg)

if __name__ == "__main__":
    hydra_main()


