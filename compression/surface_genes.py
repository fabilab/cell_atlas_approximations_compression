# vim: fdm=indent
'''
author:     Fabio Zanini
date:       06/06/24
content:    Identify surface proteins (genes) using gene ontology and other methods.
'''
import os
import pathlib
import numpy as np
import h5py
from goatools.anno.gaf_reader import GafReader
from goatools.obo_parser import OBOReader


go_folder = pathlib.Path('../data/gene_ontology')

go_association_file_dict = {
    'h_sapiens': go_folder / 'goa_human.gaf',
    'm_musculus': go_folder / 'mouse_mgi.gaf',
}


if __name__ == "__main__":

    print('Identify GO terms for cell surface/membrane')
    if False:
        surface_gos = []
        search_terms = [
            'plasma membrane',
            'cell surface',
        ]
        goa = OBOReader(
            go_folder / 'go-basic.obo',
        )
        for entry in obo:
            for search_term in search_terms:
                if (search_term in entry.name) and (entry.namespace == 'cellular_component'):
                    print(entry)
                    # NOTE: in theory we can just run this. In practice, there are a lot of odd GO terms to manual checking is good
                    # surface_gos.append(entry)
    else:
        # NOTE: Obviously, this is not ideal. Good enough to get started without too many false positives/negatives
        surface_gos = [
            'GO:0009986', # cell surface (~500 genes in human)
            'GO:0009897', # plasma membrane (another ~150 genes in human)
        ]

    surface_genes_dict = {}
    for species, gaf_file in go_association_file_dict.items():
        print(f'Identify cell surface genes in {species}')
        reader = GafReader(gaf_file)
        surface_genes = []
        for asso in reader.get_associations():
            if ('located_in' in asso.Qualifier) and (asso.GO_ID in surface_gos):
                surface_genes.append(asso.DB_Symbol)
        surface_genes = np.sort(list(set(surface_genes)))

        surface_genes_dict[species] = surface_genes

    print('Write output to h5 file')
    fn_out = pathlib.Path('../../cell_atlas_approximations_API/web/static/surface_genes/surface_genes.h5')
    folder_out = fn_out.parent
    if not folder_out.exists():
        os.mkdir(folder_out)
    with h5py.File(fn_out, 'w') as h5:
        for species, surface_genes in surface_genes_dict.items():
            h5.create_dataset(
                species, data=surface_genes.astype('S'),
            )
