import tensorflow as tf
import numpy as np
import json
from sklearn.preprocessing import StandardScaler

max_tasks = 5
max_time_limit = 30 

task_mapping = {
    "Parked in Observation Zone": 3,
    "Level 1 Ascent": 3,
    "Sample in Net Zone": 2,
    "Sample in Low Basket": 4,
    "Deliver to High Basket": 8,
    "Specimen on Low Chamber": 6,
    "Specimen on High Chamber": 6,
}

task_points = {
    "Parked in Observation Zone": 3,
    "Level 1 Ascent": 3,
    "Sample in Net Zone": 2,
    "Sample in Low Basket": 4,
    "Deliver to High Basket": 8,
    "Specimen on Low Chamber": 6,
    "Specimen on High Chamber": 6,
}

#TODO: Ensure that these are accurtae times, requres testing
# Task times mapping
task_times = {
    "Parked in Observation Zone": 5,
    "Level 1 Ascent": 5,
    "Sample in Net Zone": 8,
    "Sample in Low Basket": 10,
    "Deliver to High Basket": 12,
    "Navigate Obstacle": 6,
    "Score in Target Zone": 9
}

def load_scenarios(filename="scenarios.json"):
    with open(filename, 'r') as f:
        scenarios = json.load(f)
    return scenarios

def prepare_data(scenarios, max_tasks=5):
    features = []
    task_orders = []

    for scenario in scenarios:
        scenario_tasks = scenario['tasksPerformed']
        scenario_features = []

        # Extract task points and time as features
        for task in scenario_tasks[:max_tasks]:
            scenario_features.append(task['points'])
            scenario_features.append(task['time_required'])

        while len(scenario_features) < max_tasks * 2: 
            scenario_features.append(0)
            scenario_features.append(0)

        features.append(scenario_features)

        task_order = [task_mapping[task['task_name']] for task in scenario_tasks[:max_tasks]]
        while len(task_order) < max_tasks:
            task_order.append(-1)
        task_orders.append(task_order)

    features = np.array(features)
    task_orders = np.array(task_orders)

    return features, task_orders

scenarios = load_scenarios()

features, task_orders = prepare_data(scenarios, max_tasks)

scaler = StandardScaler()
features = scaler.fit_transform(features).astype(np.float32)

class TaskSequenceModel(tf.Module):
    def __init__(self, max_tasks, num_tasks):
        super().__init__()
        self.w1 = tf.Variable(tf.random.normal([10, 64], dtype=tf.float32), name='w1')
        self.b1 = tf.Variable(tf.zeros([64], dtype=tf.float32), name='b1')
        self.w2 = tf.Variable(tf.random.normal([64, 32], dtype=tf.float32), name='w2')
        self.b2 = tf.Variable(tf.zeros([32], dtype=tf.float32), name='b2')
        
        # Output for task sequence prediction
        self.w3 = tf.Variable(tf.random.normal([32, max_tasks * num_tasks], dtype=tf.float32), name='w3')
        self.b3 = tf.Variable(tf.zeros([max_tasks * num_tasks], dtype=tf.float32), name='b3')

    def __call__(self, x):
        x = tf.matmul(x, self.w1) + self.b1
        x = tf.nn.relu(x)
        x = tf.matmul(x, self.w2) + self.b2
        x = tf.nn.relu(x)
        
        task_predictions = tf.matmul(x, self.w3) + self.b3
        task_predictions = tf.reshape(task_predictions, [-1, max_tasks, len(task_mapping)])
        
        return task_predictions

model = TaskSequenceModel(max_tasks=max_tasks, num_tasks=len(task_mapping))

def loss_fn(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1, y_true.shape[1]])  # Shape: (batch_size, max_tasks)
    y_pred = tf.reshape(y_pred, [-1, y_true.shape[1], len(task_mapping)])  # Shape: (batch_size, max_tasks, num_tasks)

    mask = tf.not_equal(y_true, -1)
    
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)

    task_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True))

    return task_loss

optimizer = tf.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

epochs = 50
batch_size = 16
dataset = tf.data.Dataset.from_tensor_slices((features, task_orders)).batch(batch_size)

for epoch in range(epochs):
    for batch_features, batch_labels in dataset:
        loss = train_step(model, batch_features, batch_labels)
    print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')

@tf.function(input_signature=[tf.TensorSpec(shape=[None, 10], dtype=tf.float32)])
def predict(features):
    task_predictions = model(features)
    return task_predictions

tf.saved_model.save(model, "saved_task_model", signatures={'serving_default': predict.get_concrete_function()})

converter = tf.lite.TFLiteConverter.from_saved_model("saved_task_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('task_model_optimized.tflite', 'wb') as f:
    f.write(tflite_model)

sample_features = features[:5]

def get_valid_task_sequence(features, max_time=30, max_iterations=100):
    for iteration in range(max_iterations):
        sample_predictions = model(features)

        reverse_task_mapping = {v: k for k, v in task_mapping.items()}

        predicted_tasks = []
        for task_set in sample_predictions.numpy():
            task_indices = np.argmax(task_set, axis=-1)
            task_sequence = [reverse_task_mapping.get(int(index), "Unknown Task") for index in task_indices]
            predicted_tasks.append(task_sequence)
        
        total_points = sum([task_points.get(task, 0) for task in predicted_tasks[0]])
        total_time = sum([task_times.get(task, 0) for task in predicted_tasks[0]])
        
        if total_time <= max_time:
            return predicted_tasks[0], total_points, total_time
        
        print(f"Iteration {iteration + 1}: Total time {total_time}s exceeds the limit. Retrying...")

    return predicted_tasks[0], total_points, total_time

for i in range(5):
    predicted_tasks, total_points, total_time = get_valid_task_sequence(sample_features[i].reshape(1, -1))

    print(f"\nScenario {i+1} - Predicted Task Sequence: {predicted_tasks}")
    print(f"Scenario {i+1} - Predicted Total Points: {total_points}")
    print(f"Scenario {i+1} - Predicted Total Time: {total_time} seconds")

print("\nModel has been trained, quantized, and saved as TensorFlow Lite.")
