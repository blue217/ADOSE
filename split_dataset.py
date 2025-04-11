import json
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split
import argparse


def split_phemeset(json_path, train_output, test_output, train_ratio=0.7, random_seed=42):
    # 读取原始数据
    with open(json_path, 'r', encoding='utf-8') as f:
        pheme_data = json.load(f)

    train_data = {}
    test_data = {}

    # 遍历每个事件域，按 label 进行划分
    for event, samples in pheme_data.items():
        label_groups = defaultdict(list)

        # 按 label 进行分组
        for sample in samples:
            if isinstance(sample, dict):
                label_groups[sample['label']].append(sample)
            else:
                label_groups[sample[-1]].append(sample)

        if len(label_groups[0]) < 10 or len(label_groups[1]) < 10:
            print(f"event: {event} too few samples to split so ignore")
            continue

        train_samples, test_samples = [], []
        # label_groups[0] = label_groups[0][:500]
        # label_groups[1] = label_groups[1][:500]

        # 按 label 进行 70/30 划分
        for label, items in label_groups.items():
            train_items, test_items = train_test_split(items, train_size=train_ratio, random_state=random_seed)
            train_samples.extend(train_items)
            test_samples.extend(test_items)

        train_data[event] = train_samples
        test_data[event] = test_samples
        print(f"event {event} total num = {len(train_samples)+ len(test_samples)}, detail result: "
              f"positive samples num = {len(label_groups[0])}, negative samples num = {len(label_groups[1])}")

    # 保存划分后的数据
    with open(train_output, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)  # ensure_ascii=False for chinese

    # with open(test_output, 'w', encoding='utf-8') as f:
    #     json.dump(test_data, f, indent=4, ensure_ascii=False)

    print(f"split dataset {args.path} finished!")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./dataset/pheme/phemeset.json")
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default="./dataset/pheme/")
    args = parser.parse_args()

    train_output = args.output_dir + "train_70.json"
    test_output = args.output_dir + "test_30.json"

    split_phemeset(json_path=args.path, train_output=train_output, test_output=test_output,
                   train_ratio=0.7, random_seed=args.random_seed)

'''
uasge: 
python split_dataset.py --path "./dataset/pheme/phemeset.json" --output_dir "./dataset/pheme/"
python split_dataset.py --path "./dataset/twitter/final_twitter.json" --output_dir "./dataset/twitter/"
python split_dataset.py --path "./dataset/weibo/weibo.json" --output_dir "./dataset/weibo/"

'''