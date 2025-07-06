python
import os
import re
import pickle

input_path = '/scratch2/zhawang/projects/registration/Geotransformer/data/ScanNetpp/metadata/ScanNetpp_2scanes_01_03_predator.pkl'
output_path = '/scratch2/zhawang/projects/registration/Geotransformer/data/ScanNetpp/metadata/ScanNetpp_2scanes_01_03.pkl'

with open(input_path, "rb") as f:
    pkl_input = pickle.load(f)

pkl_list = []
for i in range(len(pkl_input['overlap'])):
    entry = {
        'overlap': pkl_input['overlap'][i],
        'pcd0': pkl_input['tgt'][i],
        'pcd1': pkl_input['src'][i],
        'rotation': pkl_input['rot'][i].cpu().numpy(),
        'translation': pkl_input['trans'][i].squeeze().cpu().numpy(),
    }
    scene_name = os.path.basename(os.path.dirname(entry['pcd0']))
    frag_id0 = int(re.findall(r'\d+', os.path.basename(entry['pcd0']))[0])
    frag_id1 = int(re.findall(r'\d+', os.path.basename(entry['pcd1']))[0])
    entry['scene_name'] = scene_name
    entry['frag_id0'] = frag_id0
    entry['frag_id1'] = frag_id1
    pkl_list.append(entry)

with open(output_path, "wb") as f:
    pickle.dump(pkl_list, f)