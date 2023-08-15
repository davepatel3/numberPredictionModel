#!/usr/bin/env python
# coding: utf-8

# In[51]:


import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scikeras.wrappers import KerasClassifier, KerasRegressor
import gradio as gd


# In[52]:


root_logdir = os.path.join(os.curdir, "my_logs")


# In[53]:


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
run_logdir


# In[54]:


(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()


# In[55]:


X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]


# In[56]:


X_valid.shape


# In[57]:


y_valid.shape


# In[58]:


X_train.shape


# In[59]:


y_train.shape


# In[60]:


X_test.shape


# In[61]:


y_test.shape


# In[62]:


#Just some code showing 
fig, axes = plt.subplots(5, 10, figsize=(10, 5))
axes = axes.flatten()
for i in range(50):
    ax = axes[i]
    ax.imshow(X_train[i], cmap='binary')
    ax.axis('off')  # Turn off axis labels
    ax.set_title(str(y_train[i]))
    
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()


# In[63]:


X_train_full[0].shape


# In[64]:


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])


# In[65]:


#You can see that the model weights are randomly initialized to introduce nonlinearity into the data 
model.layers[1].get_weights()


# In[66]:


X_train.shape


# In[67]:


y_train.shape


# In[68]:


y_train = y_train.reshape(-1, 1)  # Reshape y_train to (num_samples, 1)
y_valid = y_valid.reshape(-1, 1)


# In[69]:


K = keras.backend

class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []
    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.learning_rate))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.learning_rate, self.model.optimizer.learning_rate * self.factor)


# In[70]:


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


# In[71]:


model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(learning_rate=1e-3),
              metrics=["accuracy"])
expon_lr = ExponentialLearningRate(factor=1.005)


# In[72]:


history = model.fit(X_train, y_train, epochs=1,
                    validation_data=(X_valid, y_valid),
                    callbacks=[expon_lr])


# In[73]:


plt.plot(expon_lr.rates, expon_lr.losses)
plt.gca().set_xscale('log')
plt.hlines(min(expon_lr.losses), min(expon_lr.rates), max(expon_lr.rates))
plt.axis([min(expon_lr.rates), max(expon_lr.rates), 0, expon_lr.losses[0]])
plt.grid()
plt.xlabel("Learning rate")
plt.ylabel("Loss")


# In[74]:


#optimal learning rate is 0.3. Optimal strat to find best learning rate is by defining a custom call
#back, that at the end of every batch, you append the loss and learning rate using the keras backend, 
#and then increase the learning rate by the passed in factor 
keras.backend.clear_session()


# In[75]:


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])


# In[76]:


model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(learning_rate=0.3),
              metrics=["accuracy"])


# In[77]:


es = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
cb = keras.callbacks.ModelCheckpoint("mnist_model_V1")


# In[78]:


model=keras.models.load_model("mnist_model_V1")


# In[79]:


model.evaluate(X_test, y_test)


# In[80]:


tcb = keras.callbacks.TensorBoard(run_logdir)


# In[81]:


history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid), callbacks=[es, cb, tcb])


# In[84]:


def predict_sketch(sketch):
    if np.sum(sketch) == 0:
        # Return placeholder labels when the sketchpad is cleared
        return {f"Class {i}": "N/A" for i in range(5)}
    
    try:
        # Preprocess the sketch (e.g., resize, normalize)
        sketch_array = np.array(sketch).reshape(-1, 28, 28, 1)
        
        # Get class probabilities from the model
        class_probs = model.predict(sketch_array)[0]
        
        # Get the top 5 class probabilities and their corresponding labels
        top_indices = class_probs.argsort()[-5:][::-1]
        top_probs = [class_probs[i] for i in top_indices]
        
        return {
            f"Class {i}": f"{prob:.2f}" for i, prob in zip(top_indices, top_probs)
        }
    except:
        # Handle unexpected errors gracefully
        return {f"Class {i}": "Error" for i in range(5)}


# In[87]:


gd.Interface(fn=predict_sketch,
             inputs="sketchpad",
             outputs="label",
             live=True).launch(share=True)


# In[ ]:





# In[ ]:





# In[ ]:




