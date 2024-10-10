import pandas as pd
import os
import rdkit.rdBase as rdb
from rdkit import Chem

# Suppress RDKit warnings
rdb.DisableLog('rdApp.warning')

def parse_poses(file, score_type='gnina'):
    poses = Chem.SDMolSupplier(file, True)
    ligands_set = file.split("/")[-1].split(".")[0]

    # Initialize variables for tracking best scores
    last_name = None
    last_smiles = ""
    res = []

    # Initialize best scores to appropriate extreme values
    best_scores = {
        'affinity': float('inf'),
        'CNNscore': -float('inf'),
        'CNNaffinity': 0,
        'CNN_VS': 0,
        'CNNaffinity_variance': 0
    }

    for p in poses:
        if p is None:
            continue  # Skip any invalid molecules

        name = p.GetProp('_Name')
        if last_name is not None and last_name != name:
            # Append the results for the last pose
            res.append((
                last_name,
                best_scores['affinity'],
                best_scores['CNNscore'],
                best_scores['CNNaffinity'],
                best_scores['CNN_VS'],
                best_scores['CNNaffinity_variance'],
                ligands_set,
                last_smiles
            ))
            # Reset best scores for the new pose
            best_scores = {
                'affinity': float('inf'),
                'CNNscore': -float('inf'),
                'CNNaffinity': 0,
                'CNN_VS': 0,
                'CNNaffinity_variance': 0
            }

        # Update tracking variables
        last_name = name
        last_smiles = Chem.MolToSmiles(p)

        # Extract properties
        affinity = float(p.GetProp('minimizedAffinity'))
        best_scores['affinity'] = min(best_scores['affinity'], affinity)

        if score_type == 'gnina':
            CNNscore = float(p.GetProp('CNNscore'))
            CNNaffinity = float(p.GetProp('CNNaffinity'))
            CNN_VS = float(p.GetProp('CNN_VS'))
            CNNaffinity_variance = float(p.GetProp('CNNaffinity_variance'))

            # Update best scores if current CNNscore is higher
            if CNNscore > best_scores['CNNscore']:
                best_scores.update({
                    'CNNscore': CNNscore,
                    'CNNaffinity': CNNaffinity,
                    'CNN_VS': CNN_VS,
                    'CNNaffinity_variance': CNNaffinity_variance
                })

    # Append the final pose
    if last_name is not None:
        res.append((
            last_name,
            best_scores['affinity'],
            best_scores['CNNscore'],
            best_scores['CNNaffinity'],
            best_scores['CNN_VS'],
            best_scores['CNNaffinity_variance'],
            ligands_set,
            last_smiles
        ))

    # Define columns based on score type
    if score_type == 'gnina':
        columns = ['Name', 'Energy', 'CNNscore', 'CNNaffinity', 'CNN_VS', 'CNNaffinity_variance', 'Ligands', 'SMILES']
    else:
        columns = ['Name', 'Energy', 'Ligands', 'SMILES']

    return pd.DataFrame(res, columns=columns)

def parse_all_files_in_folder(root_dir, score_type='gnina'):
    res = []
    for path, subdirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('sdf'):
                try:
                    r = parse_poses(os.path.join(path, file), score_type=score_type)
                    res.append(r)
                except Exception as e:
                    print("Failed to analyse: ", os.path.join(path, file), " Error: ", e)
    return pd.concat(res, ignore_index=True) if res else pd.DataFrame()

def get_gnina_scores(root_dir):
    return parse_all_files_in_folder(root_dir, score_type='gnina')

def get_smina_scores(root_dir):
    return parse_all_files_in_folder(root_dir, score_type='smina')

# Example usage
# gnina_chembl = get_gnina_scores('outputs/PARP_sep_2023/gnina/val')
# smina_chembl = get_smina_scores('outputs/PARP_sep_2023/smina/val')
