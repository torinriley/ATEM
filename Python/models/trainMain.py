import tensorflow as tf
import numpy as np
import json
from sklearn.preprocessing import StandardScaler

# load scenarios from JSON file
def load_scenarios(filename="scenarios.json"):
    with open(filename, 'r') as f:
        scenarios = json.load(f)
    return scenarios

# prepare features and labels for training
def prepare_data(scenarios, max_tasks=5):
    features = []
    labels = []

    for scenario in scenarios:
        scenario_tasks = scenario['tasksPerformed']
        scenario_features = []

        for task in scenario_tasks[:max_tasks]:
            scenario_features.append(task['points']) 
            scenario_features.append(task['time_required'])

        while len(scenario_features) < max_tasks * 2: 
            scenario_features.append(0) 
            scenario_features.append(0)

        total_points = scenario['totalPoints']
        features.append(scenario_features)
        labels.append(total_points)

    features = np.array(features)
    labels = np.array(labels)

    return features, labels

# load and process scenarios
scenarios = load_scenarios()

features, labels = prepare_data(scenarios)

scaler = StandardScaler()
features = scaler.fit_transform(features)

# define the model architecture
class TaskModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.w1 = tf.Variable(tf.random.normal([10, 64]), name='w1')
        self.b1 = tf.Variable(tf.zeros([64]), name='b1')
        self.w2 = tf.Variable(tf.random.normal([64, 32]), name='w2')
        self.b2 = tf.Variable(tf.zeros([32]), name='b2')
        self.w3 = tf.Variable(tf.random.normal([32, 1]), name='w3')
        self.b3 = tf.Variable(tf.zeros([1]), name='b3')

    def __call__(self, x):
        x = tf.matmul(x, self.w1) + self.b1
        x = tf.nn.relu(x)
        x = tf.matmul(x, self.w2) + self.b2
        x = tf.nn.relu(x)
        x = tf.matmul(x, self.w3) + self.b3
        return x

# instantiate the model
model = TaskModel()

# define loss function
def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

optimizer = tf.optimizers.Adam(learning_rate=0.001)

# define training step
@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# train the model
epochs = 50
batch_size = 16
dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)

for epoch in range(epochs):
    for batch_features, batch_labels in dataset:
        loss = train_step(model, batch_features, batch_labels)
    print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')

# save the trained model
tf.saved_model.save(model, "saved_task_model")

# convert to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_saved_model("saved_task_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# save the TensorFlow Lite model
with open('task_model_optimized.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model has been trained, quantized, and saved as TensorFlow Lite.")