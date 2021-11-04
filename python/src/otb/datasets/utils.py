
def has_package_installed(*package_names):
    import importlib
    return all(importlib.util.find_spec(package_name) for package_name in package_names)


def google_drive_download_link(identifier):
    return f'https://drive.google.com/uc?id={identifier}'

