import pathlib


root_repo_folder = pathlib.Path(__file__).parent.parent.parent

# Try to put the output in the API repo if available
output_folder = root_repo_folder / '..' / 'cell_atlas_approximations_API' / 'web' / 'static' / 'atlas_data'
if not output_folder.is_dir():
    output_folder = root_repo_folder / 'data' / 'atlas_approximations'
