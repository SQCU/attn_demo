import sys
import os
from pathlib import Path

import shutil
import zipfile
import hashlib

#also from bbc-esq
def set_cuda_paths():
    venv_base = Path(sys.executable).parent.parent
    nvidia_base_path = venv_base / 'Lib' / 'site-packages' / 'nvidia'
    cuda_path_runtime = nvidia_base_path / 'cuda_runtime' / 'bin'
    cuda_path_runtime_lib = nvidia_base_path / 'cuda_runtime' / 'bin' / 'lib' / 'x64'
    cuda_path_runtime_include = nvidia_base_path / 'cuda_runtime' / 'include'
    cublas_path = nvidia_base_path / 'cublas' / 'bin'
    cudnn_path = nvidia_base_path / 'cudnn' / 'bin'
    nvrtc_path = nvidia_base_path / 'cuda_nvrtc' / 'bin'
    nvcc_path = nvidia_base_path / 'cuda_nvcc' / 'bin'
    
    paths_to_add = [
        str(cuda_path_runtime),
        str(cuda_path_runtime_lib),
        str(cuda_path_runtime_include),
        str(cublas_path),
        str(cudnn_path),
        str(nvrtc_path),
        str(nvcc_path),
    ]
    
    current_value = os.environ.get('PATH', '')
    new_value = os.pathsep.join(paths_to_add + [current_value] if current_value else paths_to_add)
    os.environ['PATH'] = new_value
    
    # i blame triton
    triton_cuda_path = nvidia_base_path / 'cuda_runtime'
    os.environ['CUDA_PATH'] = str(triton_cuda_path)

#set_cuda_paths()

class DependencyUpdater:
    def __init__(self):
        self.site_packages_path = self.get_site_packages_path()

    def get_site_packages_path(self):
        paths = sys.path
        site_packages_paths = [Path(path) for path in paths if 'site-packages' in path.lower()]
        return site_packages_paths[0] if site_packages_paths else None

    def find_dependency_path(self, dependency_path_segments):
        current_path = self.site_packages_path
        if current_path and current_path.exists():
            for segment in dependency_path_segments:
                next_path = next((current_path / child for child in current_path.iterdir() if child.name.lower() == segment.lower()), None)
                if next_path is None:
                    return None
                current_path = next_path
            return current_path
        return None

    @staticmethod
    def hash_file(filepath):
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as afile:
            buf = afile.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def copy_and_overwrite_if_necessary(self, source_path, target_path):
        if not target_path.exists() or DependencyUpdater.hash_file(source_path) != DependencyUpdater.hash_file(target_path):
            shutil.copy(source_path, target_path)
            DependencyUpdater.print_status("SUCCESS", f"{source_path} has been successfully copied to {target_path}.")
        else:
            DependencyUpdater.print_status("SKIP", f"{target_path} is already up to date.")

    def update_file_in_dependency(self, source_folder, file_name, dependency_path_segments):
        target_path = self.find_dependency_path(dependency_path_segments)
        if target_path is None:
            self.print_status("ERROR", "Target dependency path not found.")
            return

    @staticmethod
    def print_status(status, message):
        colors = {
            "SUCCESS": "\033[92m",  # Green
            "SKIP": "\033[93m",     # Yellow
            "ERROR": "\033[91m",    # Red
            "INFO": "\033[94m"      # Blue
        }
        reset_color = "\033[0m"
        print(f"{colors.get(status, reset_color)}[{status}] {message}{reset_color}")

    @staticmethod
    def print_ascii_table(title, rows):
        table_width = max(len(title), max(len(row) for row in rows)) + 4
        border = f"+{'-' * (table_width - 2)}+"
        print(border)
        print(f"| {title.center(table_width - 4)} |")
        print(border)
        for row in rows:
            print(f"| {row.ljust(table_width - 4)} |")
        print(border)



#from bbc-esq
def add_cuda_files():
    updater = DependencyUpdater()

    updater.print_ascii_table("CUDA FILES UPDATE", ["Copying ptxas.exe", "Extracting cudart_lib.zip"])

    source_path = updater.find_dependency_path(["nvidia", "cuda_nvcc", "bin"])
    if source_path is None:
        updater.print_status("ERROR", "Source path for ptxas.exe not found.")
        return

    source_file = source_path / "ptxas.exe"
    if not source_file.exists():
        updater.print_status("ERROR", "ptxas.exe not found in the source directory.")
        return

    target_path = updater.find_dependency_path(["nvidia", "cuda_runtime", "bin"])
    if target_path is None:
        updater.print_status("ERROR", "Target path (cuda_runtime) not found.")
        return

    target_file = target_path / "ptxas.exe"
    print(source_file)
    print(target_file)
    updater.copy_and_overwrite_if_necessary(source_file, target_file)

    #zip_path = Path(__file__) / "assets" / "cuda_12.4_lib.zip"
    #if not zip_path.exists():
    #    updater.print_status("ERROR", "cudart_lib.zip not found.")
    #    return

    cuda_lib_runtime_path = target_path.parent
    if target_path is None or not target_path.exists():
        updater.print_status("ERROR", "Parent directory of cuda_runtime/bin not found.")
        return
"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cuda_lib_runtime_path)
            updater.print_status("SUCCESS", f"Extracted cudart_lib.zip to {cuda_lib_runtime_path}")
    except zipfile.BadZipFile:
        updater.print_status("ERROR", "cudart_lib.zip is corrupted or not a zip file.")
    except PermissionError:
        updater.print_status("ERROR", "Permission denied when extracting cudart_lib.zip.")
    except Exception as e:
        updater.print_status("ERROR", f"Unexpected error during extraction: {str(e)}")
        """