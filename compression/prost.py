import os
import sys
import gc
import time
import argparse
import pathlib
import glob
import h5py
import hdf5plugin
import numpy as np
from scipy.fftpack import dct, idct
import torch
from esm.data import Alphabet


def iterate_h5(h5file):
    with h5py.File(h5file) as fh:
        names = fh["gene_expression"]["features"].asstr()[:]
        for i, name in enumerate(names):
            seq = fh["gene_expression"]["feature_sequences"]["sequences"].asstr()[i]
            yield name, seq


# The following functions are PROST, straight from:
# https://github.com/MesihK/prost/blob/master/src/pyprost/prosttools.py
def iterate_fasta(fastafile):
    import gzip
    from itertools import groupby

    with gzip.open(fastafile, "rt") as fh:
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
        for header in faiter:
            header = next(header)[1:].strip()
            seq = "".join(s.strip() for s in next(faiter))
            yield header, seq


def sanitise_name(name):
    # Uniprot files have the gene name as in GN=AAA
    if "GN=" in name:
        tmp = name.split(" ")
        for i in tmp:
            if "GN=" in i:
                name = i[3:]
                break
    return name


def sanitise_sequence(seq):
    """Optional stop codon."""
    return seq.split("*")[0]


def embed_protein_sequence(seq, model, batch_converter):
    """Embed protein sequence into vector space"""

    def _embed(seq):
        # Tokenise
        # TODO: Surely there's a better way to do this?
        _, _, toks = batch_converter([("prot", seq)])
        if torch.cuda.is_available() and not args.nogpu:
            toks = toks.to(device="cuda", non_blocking=True)

        # Call ESM1b on the tokens
        results = model(toks)

        # Fetch results
        for i in range(len(results)):
            results[i] = results[i].to(device="cpu")[0].detach().numpy()
        return results

    def _embed_chunked(seq):
        embtoks = None
        l = len(seq)
        piece = int(l / 1022) + 1
        part = l / piece
        for i in range(piece):
            st = int(i * part)
            sp = int((i + 1) * part)
            results = _embed(seq[st:sp])
            if embtoks is not None:
                for i in range(len(results)):
                    embtoks[i] = np.concatenate(
                        (embtoks[i][: len(embtoks[i]) - 1], results[i][1:]), axis=0
                    )
            else:
                embtoks = results
        return embtoks

    seq = seq.upper()
    return _embed_chunked(seq) if len(seq) > 1022 else _embed(seq)


def quantise_protein_vector_2D(emb, n=5, m=44):
    """Quantise protein vector"""

    def _standard_scale(v):
        M = np.max(v)
        m = np.min(v)
        return (v - m) / float(M - m)

    def _iDCTquant(v, n):
        f = dct(v.T, type=2, norm="ortho")
        trans = idct(f[:, :n], type=2, norm="ortho")
        for i in range(len(trans)):
            trans[i] = _standard_scale(trans[i])
        return trans.T

    # First and last tokens are BoS and EoS
    # Quantise along first dimension
    idct_res = _iDCTquant(emb[1 : len(emb) - 1], n)
    # Quantise along second dimension
    idct_res = _iDCTquant(idct_res.T, m).T
    # Flatten
    idct_res = idct_res.ravel()
    # Convert 0-1 to actual uint8
    idct_res = (idct_res * 127).astype("int8")
    return idct_res


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract per-sequence representations and model outputs for sequences in multiple FASTA files",
    )
    parser.add_argument(
        "--nogpu", action="store_true", help="Do not use GPU even if available"
    )
    parser.add_argument("--species", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-sequences", type=int, default=-1)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--use-approximations", action="store_true")

    args = parser.parse_args()

    if not args.use_approximations:
        # fasta files
        input_files = glob.glob("../data/peptide_sequences/*.fasta.gz")
    else:
        # approximation files
        input_files = glob.glob("../data/atlas_approximations/*.h5")

    if args.species:
        species = args.species.split(",")
        tmp = set(input_files)
        input_files = []
        for organism in species:
            for fn in tmp:
                if organism in fn:
                    break
            else:
                raise ValueError(f"Species not found: {organism}")
            input_files.append(fn)
            tmp.remove(fn)

    # Output file
    fn_out = pathlib.Path("../data/esm/embeddings/prost_embeddings.h5")

    try:
        print("Build ESM1b model for PROST")
        model = torch.jit.load("../data/esm/models/traced_esm1b_25_13.pt").eval()
        alphabet = Alphabet.from_architecture("ESM-1b")
        batch_converter = alphabet.get_batch_converter()

        # https://stackoverflow.com/a/63616077
        # This prevents memory leak
        for param in model.parameters():
            param.grad = None
            param.requires_grad = False

        print("Transfer model to GPU")
        if torch.cuda.is_available() and not args.nogpu:
            model = model.cuda()
            print("Transferred model to GPU")
        else:
            torch._C._jit_set_profiling_mode(False)
            model = torch.jit.freeze(model)
            model = torch.jit.optimize_for_inference(model)

        for input_file in input_files:
            species = input_file.split("/")[-1].split(".")[0]
            print(species)

            with h5py.File(fn_out, "a") as h5:
                if (not args.overwrite) and species in h5:
                    print("Already exists, skipping.")
                    continue

            input_file = pathlib.Path(input_file)
            if str(input_file).endswith(".fasta.gz"):
                print("Read input sequences from FASTA.GZ file")
                sequence_iter = iterate_fasta(input_file)
            else:
                print("Read input sequences from approximation (H5) file")
                sequence_iter = iterate_h5(input_file)

            sequence_labels = []
            sequence_representations = []
            times = []
            lengths = []
            errors = []
            with torch.inference_mode():
                for i, (name, seq) in enumerate(sequence_iter):
                    print(f"Processing sequence {i + 1}", end="\r")
                    name = sanitise_name(name)
                    seq = sanitise_sequence(seq)
                    # Too short sequences make no sense anyway
                    if len(seq) < 10:
                        continue

                    try:
                        t0 = time.time()
                        # Embed in chunks if longer than 1022
                        esm_output = embed_protein_sequence(seq, model, batch_converter)
                        q25_544 = quantise_protein_vector_2D(esm_output[1], 5, 44)
                        q13_385 = quantise_protein_vector_2D(esm_output[0], 3, 85)
                        quantised = np.concatenate([q25_544, q13_385])
                        t1 = time.time()
                        if len(quantised) != 475:
                            lq = len(quantised)
                            raise ValueError(
                                f"Quantisation for {name} gave length: {lq}"
                            )
                    except:
                        print()
                        print(name, seq)
                        errors.append((name, seq))
                        raise
                    else:
                        # Accumulate outputs
                        sequence_labels.append(name)
                        sequence_representations.append(quantised)
                        lengths.append(len(seq))
                        times.append(t1 - t0)

                    if (args.max_sequences != -1) and (i + 1 >= args.max_sequences):
                        break
                print(f"All done: {i+1} sequences")

            if len(sequence_representations) == 0:
                print("No sequences, skipping")
                continue

            sequence_representations = np.vstack(sequence_representations)

            if not args.no_save:
                print("Store to file")
                comp_kwargs = hdf5plugin.Zstd(clevel=22)
                with h5py.File(fn_out, "a") as h5:
                    if species in h5:
                        h5.pop(species)
                    speciesg = h5.create_group(species)
                    speciesg.create_dataset(
                        "features", data=np.array(sequence_labels).astype("S")
                    )
                    speciesg.create_dataset(
                        "embeddings", data=sequence_representations, dtype="u1"
                    )

            del times, lengths, sequence_representations, sequence_labels
            gc.collect()

    except:
        print("Exception raised, removing model to clean GPU...")
        del model, alphabet, batch_converter
        gc.collect()
        print("Done, reraising exception...")
        raise
