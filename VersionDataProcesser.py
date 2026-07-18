from csv import DictReader, DictWriter
from os import listdir
from os.path import isdir, isfile, getsize
from shutil import copyfile
from typing import Literal

__all_ = ["load_version_data", "import_mca_files"]


def load_version_data(input_filename: str) -> dict[str, str]:
    data = {}
    with open(input_filename, newline="") as file:
        file_reader = DictReader(file, delimiter=",")
        fieldnames = file_reader.fieldnames
        for row in file_reader:
            data[row[fieldnames[0]]] = {
                fieldname: row[fieldname] for fieldname in fieldnames[1:]
            }
    return data


def import_mca_files(
    launcher: Literal["Prism Launcher"],
    version_data: dict[str, str],
    mc_path: str,
    out_path: str,
) -> None:
    if launcher == "Prism Launcher":
        world_path = "minecraft/saves/New World"
    else:
        raise ValueError("Unsupported/Unknown Launcher.")

    mc_path = (mc_path + "/instances").replace("//", "/")
    for version in version_data.keys():
        # Old Format
        if isdir(f"{mc_path}/{version}/{world_path}/region"):
            directory = f"{mc_path}/{version}/{world_path}/region"
            for file in listdir(directory):
                if isfile(f"{directory}/{file}"):
                    copyfile(
                        f"{directory}/{file}",
                        f"{out_path}/{file.split('.mca')[0]}-{version}.mca",
                    )
                    break
        # New Format
        elif isdir(
            f"{mc_path}/{version}/{world_path}/dimensions/minecraft/overworld/region"
        ):
            directory = f"{mc_path}/{version}/{world_path}/dimensions/minecraft/overworld/region"
            for file in listdir(directory):
                if (
                    isfile(f"{directory}/{file}")
                    and getsize(f"{directory}/{file}") != 0
                ):
                    copyfile(
                        f"{directory}/{file}",
                        f"{out_path}/{file.split('.mca')[0]}-{version}.mca",
                    )
                    break
        else:
            print(f"Save data does not exist. Skipping version {version}")


def _process_raw_version_data(
    input_filename: str, output_filename: str, fieldnames: list[str]
) -> None:
    with open(input_filename, newline="") as raw_data:
        with open(output_filename, "w", newline="") as processed_data:
            reader = DictReader(raw_data, delimiter=",")
            writer = DictWriter(
                processed_data, fieldnames=["Prism Launcher Version"] + fieldnames
            )
            writer.writeheader()

            for row in reader:
                string_replacements = {
                    "release candidate": "rc",
                    "pre-release": "pre",
                    " experimental": "_experimental",
                    " unobfuscated": "_unobfuscated",
                    "java edition ": "",
                }

                version_name = row[fieldnames[0]].lower()
                for key, value in string_replacements.items():
                    version_name = version_name.replace(key, value)
                if version_name == "1.19 deep dark_experimental snapshot 1":
                    version_name = "1.19_deep_dark_experimental_snapshot 1"
                version_name = "-".join(version_name.split(" "))

                protocol_version = row[fieldnames[1]].split("(")[0]

                writer.writerow(
                    {
                        "Prism Launcher Version": version_name,
                        fieldnames[0]: row[fieldnames[0]].replace("Java Edition ", ""),
                        fieldnames[1]: protocol_version,
                        fieldnames[2]: row[fieldnames[2]],
                        fieldnames[3]: row[fieldnames[3]],
                        fieldnames[4]: row[fieldnames[4]],
                    }
                )
