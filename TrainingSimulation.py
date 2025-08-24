# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

print("Setting UP")

from utlis import  *
from sklearn.model_selection import train_test_split

#### Step 1 ⇒ Import Data Collection
path = 'data'
data = importDataInfo(path)

#### Step 2 ⇒ Initialize Data
data = balanceData(data, display=True)

#### Step 3 ⇒ Prepare for Processing
imagesPath, steerings = loadData(path, data)
# print(imagesPath[0])
# print(steerings[0])

#### Step 4 ⇒ Split for Training and Validation
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2,random_state=10)
print('Total Training Images : ',len(xTrain))
print('Total Validation Images : ',len(xVal))

#### Step 5 ⇒ Augmentation Images(Pan, Zoom, Brightness, Flip)

#### Step 6 ⇒ PreProcessing

#### Step 7 ⇒ Batch Generator

#### Step 8 ⇒ Create Model
model = creatModel()
model.summary()

batch_size = 64
#
# #### Step 9 ⇒ Training Model
history = model.fit(batchGen(xTrain, yTrain, 100,1), steps_per_epoch=300, epochs=10,
          validation_data=batchGen(xVal, yVal, 100, 0), validation_steps=200)
#
# #### Step 10 ⇒ Saving And Plot
model.save('model.h5')
print('Model Saved')


plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0, 1])
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.savefig("loss_curve.png")
print("Training curve saved as loss_curve.png")


