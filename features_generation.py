from load_mol2_data import *
import os
from signature_feature import *
from load_6angstroms_data import *


"""load Kd data"""
dir = "./Pair_coupling_dataset/LUT620CLA612_FRAMES/"
dirlist = os.listdir(dir)
dirlist = [item for item in dirlist if item[-5:] == '.mol2']
file_dir = dir + dirlist[0]

mol = load_atom(file_dir)  # a dictionary contains molecules information and all atom's information

#activity=np.load('activity.npy')

index_map = index_KdKi_map(dir)
L = 3
cat_dim =len(index_map)
deg_sig = 3
adj = load_adjacent_matrix(file_dir, mol)



"""the feature set on molecule level, output a dictionary of expected signature and log-signature on categorical path
and coordinate path"""
features=[]
for i in range(len(dirlist)):
    print(i)
    file_dir = dir + dirlist[i]
    mol_features = mol_KdKi_features(file_dir, index_map, L, cat_dim, deg_sig, 'sig')
    features.append(np.concatenate([mol_features['expected_cat_sig'],mol_features['expected_xyz_sig']],axis=1))

dir = "6 Angstroms/database_Kd_6A/"
dirlist = os.listdir(dir)
dirlist = [item for item in dirlist if item[-4:] == '.txt']
adj_list=[]
len_list=[]
activity=[]
for i in range(len(dirlist)):
    print(i)
    file_dir = dir + dirlist[i]
    mol = load_KdKi_data(file_dir)
    len_list.append(mol['mol_info'].num_atoms)
    adj_list.append(load_KdKi_adj(file_dir, mol))
    activity.append(mol['mol_info'].activity)

def padding_zeros(feature_list,adj_list,len_list):
    max_len=max(len_list)
    pad_features=np.zeros([max_len,feature_list[0].shape[1]])
    pad_feature_list=[]
    pad_adj=np.zeros([max_len,max_len])
    pad_adj_list=[]
    for i in range(len(len_list)):
        print(i)
        pad_features[:len_list[i],:]=feature_list[i]
        pad_feature_list.append(pad_features)
        pad_adj[:len_list[i],:len_list[i]]=adj_list[i]
        pad_adj_list.append(pad_adj)
    return np.array(pad_feature_list),np.array(pad_adj_list)

features_mat,adj_mat=padding_zeros(features,adj_list,len_list)

np.save('features.npy',features_mat)
np.save('adj.npy',adj_mat)
np.save('length.npy',np.array(len_list))
np.save('activity.npy',activity)
features_1=np.load("features.npy")





'''compute original features, xyz data and one hot encoded atom type information'''
index_map = index_KdKi_map(dir)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(list(index_map.keys()))
original_features=[]
for i in range(len(dirlist)):
    print(i)
    file_dir = dir + dirlist[i]
    mol = load_KdKi_data(file_dir)
    temp=[]
    for j in range(mol['mol_info'].num_atoms):
        temp.append(np.concatenate([mol['atoms'][j].features,le.transform([mol['atoms'][j].ntype])]))
    original_features.append(np.array(temp))
original_features1=np.array(original_features)
np.save('ori_features.npy',np.array(original_features))
original_features1=np.load('ori_features.npy',allow_pickle="True")

def padding_zeros1(feature_list,len_list):
    max_len=max(len_list)
    pad_features=np.zeros([max_len,feature_list[0].shape[1]])
    pad_feature_list=[]


    for i in range(len(len_list)):
        print(i)
        pad_features[:len_list[i],:]=feature_list[i]
        pad_feature_list.append(pad_features)

    return np.array(pad_feature_list)

input_features=padding_zeros1(original_features,len_list)
np.save('original_features.npy',input_features)
original_features1=np.load('original_features.npy')
input_features.shape