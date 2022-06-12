import os


def get_path(*sub_dir):
    return os.path.join(os.path.dirname(__file__), *sub_dir)


def get_parent_path():
    return os.path.dirname(get_path())


def get_data_path(*sub_dir):
    return os.path.join(get_path('data'), *sub_dir)


def get_images_path(*sub_dir):
    return os.path.join(get_path('images'), *sub_dir)
