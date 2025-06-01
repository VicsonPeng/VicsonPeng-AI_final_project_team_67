import os
import csv

skeleton_dir    = "skeletons"
annot_train     = "Annot_TrainList.txt"
annot_test      = "Annot_TestList.txt"
output_base_dir = "gestures"  # 根目錄

def split_and_export(annot_file, split_name):
    print(f"\n── 處理 {annot_file} ──")

    with open(annot_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # row = [video, label, id, t_start, t_end, frames]
            if len(row) < 6:
                print(f"[警告] 格式錯誤，跳過：{row}")
                continue

            video   = row[0].strip()
            label   = row[1].strip()
            t_start = int(row[3].strip())
            t_end   = int(row[4].strip())

            # skeleton 路徑
            skel_path = os.path.join(skeleton_dir, f"{video}.txt")
            if not os.path.isfile(skel_path):
                print(f"[警告] 找不到 {skel_path}，跳過此筆")
                continue

            with open(skel_path, "r") as skf:
                all_frames = skf.readlines()

            # 正確切片，注意 Python index 從 0 開始，t_end 為 inclusive
            frames = [line.strip() for line in all_frames[t_start - 1 : t_end]]

            # 輸出資料夾：gestures/train/G01/
            out_dir = os.path.join(output_base_dir, split_name, label)
            os.makedirs(out_dir, exist_ok=True)

            # 輸出檔名
            out_filename = f"{video}_{t_start}_{t_end}.txt"
            out_path = os.path.join(out_dir, out_filename)

            # 寫入該段座標
            with open(out_path, "w") as out_f:
                out_f.write("\n".join(frames))
                out_f.write("\n")

    print(f"✅ 完成 {split_name}：輸出至 {os.path.join(output_base_dir, split_name)}")

# 執行兩次：train + test
split_and_export(annot_train, "train")
split_and_export(annot_test,  "test")
