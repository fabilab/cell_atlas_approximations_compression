import pathlib
import platform


root_repo_folder = pathlib.Path(__file__).parent.parent.parent

# Try to put the output in the API repo if available
output_folder = root_repo_folder / '..' / 'cell_atlas_approximations_API' / 'web' / 'static' / 'atlas_data'
if not output_folder.is_dir():
    output_folder = root_repo_folder / 'data' / 'atlas_approximations'


if platform.node() == 'archfabilab1':
    raw_atlas_folder = pathlib.Path('/mnt/data/projects/cell_atlas_approximations/reference_atlases/') / 'data' / 'raw_atlases'
    curated_atlas_folder = pathlib.Path('/mnt/data/projects/cell_atlas_approximations/reference_atlases/') / 'data' / 'curated_atlases'
else:
    raw_atlas_folder = root_repo_folder / 'data' / 'raw_atlases'
    curated_atlas_folder = root_repo_folder / 'data' / 'curated_atlases'

