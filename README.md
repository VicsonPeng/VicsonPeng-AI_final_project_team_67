# AI final project team67 | Gesture Recognization
### **Dataset Download**  
download ipn dataset with the link (provided by ipn website) https://drive.google.com/drive/folders/1aL645mUzzAvoTMwJKrbtQiNiDVJZ2EsA

### **Train the models**  
If you want to run HTconvNet3d 
```
cd HTconvNet3D
```

If you want to run HTconvNet2D 
```
cd HTconvNet2D
```

### **Process datasets**  
Use the preprocessing code in the **data processing** folder to process the data and put them in the data folder.

### **Training**  
```
python train.py --batch-size 512 --epochs 600 --dataset 1 --lr 0.001 | tee train.log
```

### **Testing**
```
python test.py --model-path ibn_experiment/timestamp/model.pt --dataset 1
```
note that model.pt can be found in ibn_experiments folder after training
