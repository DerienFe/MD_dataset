#!/usr/bin/env python
# coding: utf-8

# In[11]:


"""
This script provides functions load standard mol2 files as diretories.
in which contains:
    'atom': atom type, coordinates, atom index
    'mol info': activity/coupling/Hamiltonian(label), number of total atoms, summary of atom types.

"""


# In[21]:


import numpy as np
import os
from molecule_lib import Node, Mol


# In[16]:


def load_standardmol2(db_dir):    #db_dir is the file path. 
    data_in_file =[]
    with open (db_dir, "r") as current:
        for row in current:
            line = row.split()
            data_in_file.append(line)
    atom_start = data_in_file.index(['@<TRIPOS>ATOM']) + 1
    atom_end = data_in_file.index(['@<TRIPOS>BOND'])
    atom_lines= data_in_file[atom_start:atom_end]
    atoms =[]    #This will be a list of "atom object", as defined in molecule_lib.py file.
    atom_type_list = []    #This is a list of existing atom types found in the loaded file. useful when later summarizing all atom types in total.

    for line in atom_lines:
        atom_type = line[1][0]    #Atom type is clustered as C,H,N,O etc, no subtypes like CA, CB.
        x_y_z = np.asarray([line[2:5]], float)
        num_atoms = line[0]    #Constantly update var num_atoms, the last value(index) is the total number of atoms.
        atoms.append(Node(atom_type, x_y_z, num_atoms))    #append this line as a atom object to the list "atoms"
        if atom_type not in atom_type_list:
            atom_type_list.append(atom_type)
        mol={'mol_info':Mol(num_atoms, atom_type_list),'atoms':atoms}     
    return mol


"""
note instead of returning idx, atom_list, activity in keyword 'mol_info', this function only return index, atom_type_list
"""


# In[19]:


#db_dir = "Pair_coupling_dataset\LUT620CLA612_FRAMES\lut620cla612.1.mol2"


# In[15]:


#load_standardmol2(db_dir)


# In[17]:


def load_standardmol2_adj(db_dir,mol):
    data_in_file =[]
    with open (db_dir, "r") as current:
        for row in current:
            line = row.split()
            data_in_file.append(line)
    bond_start = mol2_file.index(['@<TRIPOS>BOND']) + 1
    bond_end = mol2_file.index(['@<TRIPOS>SUBSTRUCTURE'])
    bond_info=mol2_file[bond_start:bond_end]
    adjacent_matrix=np.zeros([mol['mol_info'].num_atoms,mol['mol_info'].num_atoms])
    for line in bond_info:
        adjacent_matrix[int(line[1])-1, int(line[2])-1] = 1
        adjacent_matrix[int(line[2])-1, int(line[1])-1] = 1
    return adjacent_matrix


# In[32]:


def index_map(dir):
    """

    :param dir: the directory contains all mol2 files
    :return:  the index map
    """
    dirlist = os.listdir(dir)
    #print(dirlist)
    mol2_dirlist= [item for item in dirlist if item[-4:] == 'mol2']
    all_atom_types = []
    for file_dir in mol2_dirlist:
        #print(file_dir)
        mol = load_standardmol2(os.path.join(dir,file_dir))
        for atom in mol['mol_info'].atom_list:
            if atom not in all_atom_types:
               all_atom_types.append(atom)
    print("atoms include are", all_atom_types)
    index_map = {}
    i = 0
    for atom in sorted(all_atom_types):
        index_map[atom] = i
        i += 1
    return index_map

"""
This function is used to determine how many atom types are here in the given directory.
"""


# In[35]:


#dir = "Pair_coupling_dataset/LUT620CLA612_FRAMES/"


# In[36]:


#index_map(dir)


# In[ ]:




