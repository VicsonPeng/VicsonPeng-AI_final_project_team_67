import os

folder_path = os.path.dirname(os.path.abspath(__file__))  # ibn_dataset/get_train_gesture.py所在的資料夾
train_gesture_path = os.path.join(folder_path, 'train')   # ibn_dataset/train

with open(os.path.join(train_gesture_path, 'train_gestures.txt'), 'w', encoding='utf-8') as wf:
    for name in os.listdir(train_gesture_path):
        full_path = os.path.join(train_gesture_path, name)
        if os.path.isdir(full_path):
            i = 1
            for filename in os.listdir(full_path):
                wf.write(name + ' ')
                old_name = os.path.join(full_path, filename)
                new_name = os.path.join(full_path, 'c' + str(i) + '.txt')
                os.rename(old_name, new_name)
                wf.write(str(i) + ' ')
                i += 1
                with open(new_name, 'r', encoding='utf-8') as rf:
                    lines = rf.readlines()
                    line_count = len(lines)
                    wf.write(str(line_count))
                    content = ''.join(lines).replace(';', ' ')
                with open(new_name, 'w', encoding='utf-8') as wrf:
                    wrf.write(content)
                wf.write('\n')
