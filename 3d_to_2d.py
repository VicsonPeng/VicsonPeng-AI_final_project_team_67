import os

def convert_all_3d_to_2d(root_dir):
    for split in ["train", "test"]:
        split_dir = os.path.join(root_dir, split)
        for label in os.listdir(split_dir):
            label_dir = os.path.join(split_dir, label)
            if not os.path.isdir(label_dir):
                continue

            for filename in os.listdir(label_dir):
                if filename.endswith(".txt"):
                    input_path  = os.path.join(label_dir, filename)
                    output_path = os.path.join(label_dir, filename)  # 可改成其他路徑避免覆蓋

                    convert_3d_to_2d(input_path, output_path)
                    print(f"✔ 已轉換 {input_path}")

# 呼叫
convert_all_3d_to_2d("gestures")
