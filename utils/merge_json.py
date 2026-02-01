import os
import json

data_root         = '/dataset/shared/magiclidar'
source_dir        = 'meta_data_v10'
merged_dir        = 'meta_data_v10_250423_appearance_added/meta_data_v10_250423_appearance_added'
updated_merge_dir = 'meta_data_v10_updated'
splits            = ['train', 'test']

for split in splits:
    src_base     = os.path.join(data_root, source_dir, split)
    merged_base  = os.path.join(data_root, merged_dir,  split)
    updated_base = os.path.join(updated_merge_dir, split)
    os.makedirs(updated_base, exist_ok=True)  # 保证输出目录存在

    for fname in os.listdir(src_base):
        src_file     = os.path.join(src_base,    fname)
        merged_file  = os.path.join(merged_base, fname)
        updated_file = os.path.join(updated_base, fname)

        # 读两个文件
        with open(src_file,   'r', encoding='utf-8') as f:
            source_meta = json.load(f)
        with open(merged_file,'r', encoding='utf-8') as f:
            merged_meta = json.load(f)

        # 用 dict map 加速查找
        src_map = { item['id']: item for item in source_meta }

        # data_list = []
        # 更新 merged_meta
        for item in merged_meta:
            src = src_map.get(item['id'])
            if src:
                item['attributes'] = src['attributes']
                item['captions']    = src['captions']
                # data_list.append(item)
            else:
                print(f"Warning: {item['id']} not found in source_meta")
        # 写到 updated_file
        # 如果你不需要缩进，可以去掉 indent 参数，加快写入速度
        with open(updated_file, 'w') as f:
            json.dump(merged_meta, f, indent=4)
