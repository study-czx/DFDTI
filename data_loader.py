import numpy as np
import pandas as pd
import funcs

feature_path_dr = "./feature/drug/"
feature_path_p = "./feature/protein/"
sim_feature_path_dr = "./feature_sim/drug/"
sim_feature_path_p = "./feature_sim/protein/"

def Get_finger():
    MACCS = np.loadtxt(feature_path_dr + "MACCS.csv", dtype=float, delimiter=",")
    Pubchem = np.loadtxt(feature_path_dr + "Pubchem.csv", dtype=float, delimiter=",")
    RDK = np.loadtxt(feature_path_dr + "RDK.csv", dtype=float, delimiter=",")
    ECFP4 = np.loadtxt(feature_path_dr + "ECFP4.csv", dtype=float, delimiter=",")
    FCFP4 = np.loadtxt(feature_path_dr + "FCFP4.csv", dtype=float, delimiter=",")
    Dr_finger = {'maccs':MACCS,'pubchem': Pubchem, 'rdk': RDK, 'ecfp4': ECFP4, 'fcfp4': FCFP4}
    return Dr_finger

def Get_seq():
    TPC = np.loadtxt(feature_path_p + "TPC.csv", dtype=float, delimiter=",", skiprows=0)
    PAAC = np.loadtxt(feature_path_p + "PAAC.csv", dtype=float, delimiter=",", skiprows=0)
    KSCTriad = np.loadtxt(feature_path_p + "KSCTriad.csv", dtype=float, delimiter=",", skiprows=0)
    CKSAAP = np.loadtxt(feature_path_p + "CKSAAP.csv", dtype=float, delimiter=",", skiprows=0)

    CTDC = np.loadtxt(feature_path_p + "CTDC.csv", dtype=float, delimiter=",", skiprows=0)
    CTDT = np.loadtxt(feature_path_p + "CTDT.csv", dtype=float, delimiter=",", skiprows=0)
    CTDD = np.loadtxt(feature_path_p + "CTDD.csv", dtype=float, delimiter=",", skiprows=0)
    CTD = np.concatenate((CTDC, CTDT, CTDD/100), axis=1)

    # CKSAAGP = np.loadtxt(feature_path_p + "CKSAAGP.csv", dtype=float, delimiter=",", skiprows=0)
    P_seq = {'PAAC': PAAC, 'KSCTriad': KSCTriad,  'TPC': TPC, 'CKSAAP':CKSAAP, 'CTD': CTD}
    return P_seq


def Get_id():
    Drug_structure = pd.read_csv("./origin_data/drug_structure_1520.csv", sep=',', dtype=str)
    Protein_seq = pd.read_csv(r"./origin_data/protein_seq_1771.csv", sep=',', dtype=str)
    Drug_id, Protein_id = [], []
    for i in range(len(Drug_structure)):
        Drug_id.append(Drug_structure['drugbank'][i])
    for j in range(len(Protein_seq)):
        Protein_id.append(Protein_seq['uniprot'][j])
    return Drug_id, Protein_id


def Get_String():
    Drug_structure = pd.read_csv("./origin_data/drug_structure_1520.csv", sep=',', dtype=str)
    Protein_seq = pd.read_csv(r"./origin_data/protein_seq_1771.csv", sep=',', dtype=str)
    SMILES, Seq = [], []
    for i in range(len(Drug_structure)):
        SMILES.append(Drug_structure['SMILES'][i])
    for j in range(len(Protein_seq)):
        Seq.append(Protein_seq['seq'][j])
    return SMILES, Seq

