import gdown
import os
import shutil
from pathlib import Path
import dotenv

def download_drive_folder(folder_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"Downloading folder: {url} -> {output_dir}")
    gdown.download_folder(url=url, output=output_dir, quiet=False, use_cookies=False)
    print(f"Finished downloading folder into {output_dir}\n")


def merge_folders(source_pattern, destination_folder):
    destination = Path(destination_folder)
    destination.mkdir(parents=True, exist_ok=True)

    source_folders = sorted(Path('.').glob(source_pattern))

    for folder in source_folders:
        for file in folder.rglob('*'):
            if file.is_file():
                relative_path = file.relative_to(folder)
                target_path = destination / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)

                if target_path.exists():
                    stem, suffix = target_path.stem, target_path.suffix
                    counter = 1
                    while True:
                        new_target = target_path.parent / f"{stem}_{counter}{suffix}"
                        if not new_target.exists():
                            target_path = new_target
                            break
                        counter += 1

                shutil.move(str(file), str(target_path))
                
    for folder in source_folders:
        os.removedirs(folder)

    print(f"All folders merged into: {destination_folder}\n")


if __name__ == "__main__":
    dotenv.load_dotenv()
    
    VIDEO_FOLDERS_IDS = [os.getenv(f'V_FOLDER_{id}') for id in range(1, 9)]
    LABEL_FOLDERS_IDS = [os.getenv(f'L_FOLDER_{id}') for id in range(1, 9)]

    output_dir_videos = "data/raw/videos2"
    output_dir_labels = "data/raw/labels2"

    for idx, id in enumerate(VIDEO_FOLDERS_IDS):
        download_drive_folder(id, output_dir_videos + f'_{idx}')
        
    for idx, id in enumerate(LABEL_FOLDERS_IDS):
        download_drive_folder(id, output_dir_labels + f'_{idx}')
    
    # these methods move the elements from folders and after everything is moved -> removes the downloaded empty folders
    merge_folders(f'{output_dir_videos}_*', output_dir_videos)
    merge_folders(f'{output_dir_labels}_*', output_dir_labels)