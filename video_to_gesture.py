import os
import csv

skeleton_dir    = "skeletons"
annot_train     = "Annot_TrainList.txt"
annot_test      = "Annot_TestList.txt"
output_base_dir = "gestures"

def split_and_export(annot_file, split_name):
    print(f"\n── 處理 {annot_file} ──")

    with open(annot_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                print(f"格式錯誤，跳過：{row}")
                continue

            video   = row[0].strip()
            label   = row[1].strip()
            t_start = int(row[3].strip())
            t_end   = int(row[4].strip())

            skel_path = os.path.join(skeleton_dir, f"{video}.txt")
            if not os.path.isfile(skel_path):
                print(f"找不到 {skel_path}，跳過此筆")
                continue

            with open(skel_path, "r") as skf:
                all_frames = skf.readlines()

            frames = [line.strip() for line in all_frames[t_start - 1 : t_end]]

            out_dir = os.path.join(output_base_dir, split_name, label)
            os.makedirs(out_dir, exist_ok=True)

            out_filename = f"{video}_{t_start}_{t_end}.txt"
            out_path = os.path.join(out_dir, out_filename)

            with open(out_path, "w") as out_f:
                out_f.write("\n".join(frames))
                out_f.write("\n")

    print(f"完成 {split_name}：輸出至 {os.path.join(output_base_dir, split_name)}")

split_and_export(annot_train, "train")
split_and_export(annot_test,  "test")
