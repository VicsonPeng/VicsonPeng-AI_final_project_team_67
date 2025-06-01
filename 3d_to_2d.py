import os

def convert_3d_to_2d_line(line):
    vals = [v for v in line.strip().split(";") if v != ""]
    if len(vals) != 63:
        return None
    xy = [v for i, v in enumerate(vals) if i % 3 != 2]
    return " ".join(xy)

def process_split(input_root, output_root):
    for label in os.listdir(input_root):
        in_label_dir = os.path.join(input_root, label)
        if not os.path.isdir(in_label_dir):
            continue

        out_label_dir = os.path.join(output_root, label)
        os.makedirs(out_label_dir, exist_ok=True)

        for fn in os.listdir(in_label_dir):
            if not fn.endswith(".txt") or "_2d" in fn:
                continue

            in_path  = os.path.join(in_label_dir, fn)
            out_path = os.path.join(out_label_dir, fn)

            with open(in_path, "r") as rf, open(out_path, "w") as wf:
                for line in rf:
                    converted = convert_3d_to_2d_line(line)
                    if converted is not None:
                        wf.write(converted + "\n")

if __name__ == "__main__":
    for folder in ["train", "test"]:
        if os.path.isdir(folder):
            dst = folder + "_2d"
            os.makedirs(dst, exist_ok=True)
            process_split(folder, dst)
        else:
            print(f"找不到資料夾：{folder}")  
