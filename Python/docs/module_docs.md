# ScenarioManager

The ScenarioManager module is designed to manage multiple scenarios, where each scenario contains a list of tasks and tracks the total points and time spent for those tasks. The ScenarioManager class facilitates adding and retrieving scenarios, and each scenario is uniquely identified by a randomly generated UUID.

# trainMain
This TensorFlow-based model is designed to predict the optimal sequence of tasks for a given scenario, maximizing points while minimizing total time spent on the tasks. 

Key Components:

1. **Task and Scenario Representation:**
	Each task is associated with a point value and time requirement (both in seconds).
	Scenarios consist of multiple tasks, and the model is trained to learn the best combination and order of tasks to achieve the highest points within a specified time limit (30 seconds).
2. **Task Features:**
	Task features include the points and time required for each task. This data is used to train the model on the relationship between tasks, points, and the time it takes to complete them.
	Padding is used when fewer than max_tasks are provided in the training data, ensuring consistency across all inputs.
3. **Model Architecture:**
	The model consists of a sequence of fully connected layers, starting with two hidden layers (with ReLU activations) that process the task features.
	The output layer predicts a sequence of tasks from the task set. It outputs a matrix of predicted tasks where each row corresponds to a task in the sequence, and the columns represent the probability for each possible task.
4. **Training:**
	The model is trained using sparse categorical cross-entropy loss, which encourages the model to learn the correct task sequence by matching predicted tasks to the true task order.
	The model also applies a custom time constraint. During training, if the predicted task sequence exceeds the 30-second time limit, it retries the prediction until a valid sequence is found or the maximum number of iterations is reached.
5. **Prediction:**
	Once trained, the model predicts the best task sequence for new scenarios, ensuring the total time does not exceed the 30-second limit.
	The output includes both the predicted sequence of tasks, the total points the sequence would achieve, and the total time required.
6. **Export to TensorFlow Lite:**
	After training, the model is exported as a TensorFlow Lite model, allowing it to be used onboard robots along with a helper JavaScript to allow for real-time integration. The model is quantized for optimization to reduce memory and computational footprint.


# main

The `main` module, in conjunction with the `scenario_classes` module, work to allow you to enter the correct data, then dynamically format the data into a JSON file, one that can be accessed through the main machine learning model, streamlining the process for data processing, and ensuring reusability in classes, for potentially different applications. Note that `task_difficulty` is a subjective measurement, that is determiedn on a 1-5 scale. Teams are encouraged to develop a through and well defiend task diffuclty measuremnet. 
