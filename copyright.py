import os
import sys


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


root = str(os.getcwd()) + "/src/"


def scan(dir: str, ext: [str]):
    subfolders, files = [], []

    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)

    for dir in list(subfolders):
        sf, f = scan(dir, ext)
        subfolders.extend(sf)
        files.extend(f)

    return subfolders, files


_, files = scan(root, [".rs"])


def read_lines(file: str, count: int) -> [str]:
    with open(file) as f:
        return f.readlines(count)


project = os.getcwd()
is_copy_right_everywhere = True
copy_right = "Copyright (c) 2022 Kamil Konior. All rights reserved"

for file in files:
    lines = read_lines(file, 5)

    has_copy_right = False
    for line in lines:
        if copy_right in line:
            has_copy_right = True
            break

    if not has_copy_right:
        file_name = file.split("/")[-1]
        path = os.path.relpath(file, project)
        print(f"{BColors.WARNING}Add copyright to file{BColors.ENDC} '{str(file_name)}', path: '{path}'",
              file=sys.stderr)

    is_copy_right_everywhere &= has_copy_right

if is_copy_right_everywhere:
    print(f"{BColors.OKGREEN}Good job!{BColors.ENDC}")
    print("Everything is alright")
