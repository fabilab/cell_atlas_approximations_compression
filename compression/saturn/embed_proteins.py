"""Embed all proteins from all termite species using ESM.

This must be run inside the Python 3.9 esm conda environment:

source ~/miniconda3/bin/activate && conda activate esm
"""

import os
import pathlib
import subprocess as sp
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Embed all proteins from all species using ESM."
    )
    parser.add_argument("--species", default=None, help="Only process these species")
    parser.add_argument("--model", default="esm1b", choices=["esm1b", "esmc"])
    args = parser.parse_args()

    fasta_root_folder = pathlib.Path(
        "/mnt/data/projects/cell_atlas_approximations/reference_atlases/data/saturn/peptide_sequences/"
    )
    fasta_files = os.listdir(fasta_root_folder)

    if args.model == "esm1b":
        output_root_folder = fasta_root_folder.parent / "esm_embeddings"
    else:
        output_root_folder = fasta_root_folder.parent / "esmc_embeddings"

    os.makedirs(output_root_folder, exist_ok=True)

    for fasta_file in fasta_files:
        species = fasta_file.split(".")[0]
        if args.species is not None and species not in args.species:
            continue

        print(f"Processing {fasta_file}")

        fasta_file_abs_path = fasta_root_folder / fasta_file
        output_folder_abs_path = output_root_folder / f"{fasta_file}_{args.model}"
        if (args.model == "esm1b") and output_folder_abs_path.exists():
            print(f"Skipping {fasta_file}, already processed")
            continue
        else:
            os.makedirs(output_folder_abs_path, exist_ok=True)

        if args.model == "esm1b":
            script_path = (
                pathlib.Path("/home/fabio/projects/termites")
                / "software"
                / "esm"
                / "scripts"
                / "extract.py"
            )
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

        else:
            from Bio import SeqIO
            import torch
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein, LogitsConfig

            with open(fasta_file_abs_path, "rt") as handle:
                for ir, record in enumerate(SeqIO.parse(handle, "fasta")):
                    name = record.id
                    fn_out = output_folder_abs_path / f"{name}.pt"
                    if fn_out.exists():
                        print(f"{ir + 1} {name} already processed")
                        continue
                    print(ir + 1, name)

                    sequence = str(record.seq)
                    protein = ESMProtein(sequence=sequence)
                    client = ESMC.from_pretrained("esmc_300m").to("cuda")  # or "cpu"
                    protein_tensor = client.encode(protein)
                    # NOTE: we might want a specific hidden layer, like PROST does on ESM1b?
                    logits_output = client.logits(
                        protein_tensor,
                        LogitsConfig(sequence=True, return_embeddings=True),
                    )
                    torch.save(logits_output.embeddings, fn_out)
