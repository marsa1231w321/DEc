import json
import os
import zipfile


MODEL_EXTENSIONS = {".obj", ".glb", ".gltf", ".stl", ".ply", ".off", ".xyz"}
TEXTURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".tif", ".tiff"}
SOUND_EXTENSIONS = {
    ".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma",
}
ANIMATION_EXTENSIONS = {".bvh", ".anim", ".dae", ".fbx"}


def asset_category(name):
    extension = os.path.splitext(name)[1].lower()
    if extension in MODEL_EXTENSIONS:
        return "Models"
    if extension in TEXTURE_EXTENSIONS:
        return "Textures"
    if extension in SOUND_EXTENSIONS:
        return "Sounds"
    if extension in ANIMATION_EXTENSIONS:
        return "Animations"
    return "Other"


def create_asset_pack(path, files):
    entries = []
    sources = [source for source in files if os.path.isfile(source)]
    if not sources:
        raise ValueError("Select at least one existing file to create an asset pack.")
    try:
        common_root = os.path.commonpath(
            [os.path.dirname(os.path.abspath(source)) for source in sources]
        )
    except ValueError:
        common_root = None
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            relative = (
                os.path.relpath(os.path.abspath(source), common_root)
                if common_root
                else os.path.basename(source)
            )
            relative = relative.replace("\\", "/")
            directory, name = os.path.split(relative)
            entry = relative
            suffix = 2
            existing = {item["path"] for item in entries}
            while entry in existing:
                stem, extension = os.path.splitext(name)
                renamed = f"{stem}_{suffix}{extension}"
                entry = f"{directory}/{renamed}" if directory else renamed
                suffix += 1
            archive.write(source, f"assets/{entry}")
            entries.append({"path": f"assets/{entry}", "category": asset_category(name)})
        archive.writestr(
            "pack.json",
            json.dumps({"format": "DEcAssetPack", "version": 1, "assets": entries}, indent=2),
        )


def list_asset_pack(path):
    try:
        archive_handle = zipfile.ZipFile(path, "r")
    except zipfile.BadZipFile as error:
        raise ValueError("This .DE archive is not a valid DEc asset pack.") from error
    with archive_handle as archive:
        try:
            manifest = json.loads(archive.read("pack.json"))
        except (KeyError, json.JSONDecodeError) as error:
            raise ValueError("This .DE archive is not a valid DEc asset pack.") from error
        if manifest.get("format") != "DEcAssetPack" or manifest.get("version") != 1:
            raise ValueError("Unsupported DEc asset pack format.")
        archive_names = set(archive.namelist())
        assets = []
        for item in manifest.get("assets", []):
            entry = item.get("path", "")
            normalized = entry.replace("\\", "/")
            if not normalized.startswith("assets/") or ".." in normalized.split("/"):
                raise ValueError(f"Unsafe asset path in pack: {entry!r}")
            if normalized not in archive_names:
                raise ValueError(f"Asset is missing from pack: {entry}")
            assets.append(
                {
                    "path": normalized,
                    "name": os.path.basename(normalized),
                    "category": item.get("category", asset_category(normalized)),
                }
            )
        return assets


def extract_asset(path, entry, destination):
    try:
        archive_handle = zipfile.ZipFile(path, "r")
    except zipfile.BadZipFile as error:
        raise ValueError("This .DE archive is not a valid DEc asset pack.") from error
    with archive_handle as archive:
        normalized = entry.replace("\\", "/")
        if not normalized.startswith("assets/") or ".." in normalized.split("/"):
            raise ValueError(f"Unsafe asset path: {entry!r}")
        try:
            source = archive.open(normalized, "r")
        except KeyError as error:
            raise ValueError(f"Asset is missing from pack: {entry}") from error
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with source, open(destination, "wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
    return destination
