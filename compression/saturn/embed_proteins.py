"""Embed all proteins from all termite species using ESM.

This must be run inside the Python 3.9 esm conda environment:

source ~/miniconda3/bin/activate && conda activate esm
"""
import os
import pathlib
import subprocess as sp
import argparse



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Embed all proteins from all species using ESM.")
    parser.add_argument('--species', default=None, help="Only process these species")
    args = parser.parse_args()

    script_path = pathlib.Path("/home/fabio/projects/termites") / "software" / "esm" / "scripts" / "extract.py"
    fasta_root_folder = pathlib.Path("/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/saturn/peptide_sequences/")
    output_root_folder = fasta_root_folder.parent / "esm_embeddings"
    os.makedirs(output_root_folder, exist_ok=True)
    fasta_files = os.listdir(fasta_root_folder)

    for fasta_file in fasta_files:
        species = fasta_file.split(".")[0]
        if args.species is not None and species not in args.species:
            continue

        print(f"Processing {fasta_file}")

        fasta_file_abs_path = fasta_root_folder / fasta_file
        output_folder_abs_path = output_root_folder / f"{fasta_file}_esm1b"
        if output_folder_abs_path.exists():
            print(f"Skipping {fasta_file}, already processed")
            continue

        call = [
            "python",
            str(script_path),
            "esm1b_t33_650M_UR50S",
            str(fasta_file_abs_path),
            str(output_folder_abs_path),
            "--include",
            "mean",
        ]
        print(" ".join(call))
        sp.run(" ".join(call), check=True, shell=True)
