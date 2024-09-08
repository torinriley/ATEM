# About Adaptive Task Execution Model (ATEM)

ATEM is an optimization machine learning model for the autonomous stage of the competition. ATEM is trained on packages of data called "scenarios". Each package contains key information about the environment, such as tasks completed, the difficulty of tasks, time left after each task, points scored on each task, success of each task, etc. The model then optimizes the list, or order, of tasks to maximize the number of points scored in the autonomous competition stage. After training, the model uses its training to produce the optimal lists of tasks in real time during the auto stage. Note that some of the references in this paper are specific to the 2024-2025 season.

![Flow Chart](/Users/torinetheridge/Downloads/ATEM/images/flowchart.png)

# **Python Build**

## Task-Based Autonomous Scenario Prediction

### Overview
This project aims to train a machine learning model using TensorFlow to predict optimal task sequences for a robotics-based autonomous system, like those used in competitive robotics (e.g., FTC). The model learns from scenarios defined in a JSON file, each containing a series of tasks performed, along with the points scored and time taken for each task. The trained model is then exported as a TensorFlow Lite model to be used for real-time decision-making on embedded systems or mobile devices.

### Project Structure

- **`trainMain.py`**: The main script that loads the dataset, trains the TensorFlow model, and converts it to TensorFlow Lite.
- **`scenario_classes.py`**: Defines the structure for a `Scenario` and `Task` class. These classes manage the tasks performed in various scenarios.
- **`main.py`**: A script that can be used to define and create new scenarios, which are saved in a JSON file for further training or testing purposes.

### Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- scikit-learn

You can install the required dependencies using the following:

```bash
pip install tensorflow numpy scikit-learn
```


### How to Use:
---



**Preparing the Data:**

The data should be prepared in a JSON format (e.g., scenarios.json). Each scenario should define:


	•tasksPerformed: A list of tasks, where each task contains task_name, 
    points, and time_required.

	•totalPoints: The total points for the scenario.

	•totalTimeSpent: The total time spent in the scenario.

An example of the data structure:

```json
[
    {
        "scenarioNumber": 1,
        "scenarioTag": "cf6a9b0f-2dd7-438d-8de7-21ff9964102b",
        "totalPoints": 11,
        "totalTimeSpent": 28,
        "tasksPerformed": [
            {
                "task_name": "Parked in Observation Zone",
                "points": 3,
                "time_required": 5
            },
            {
                "task_name": "Level 1 Ascent",
                "points": 2,
                "time_required": 5
            },
            {
                "task_name": "Sample in Net Zone",
                "points": 2,
                "time_required": 8
            },
            {
                "task_name": "Sample in Low Basket",
                "points": 4,
                "time_required": 10
            }
        ]
    }
]
```


**Running the Model Training:**

To train the model, run the trainMain.py file. This script will:

	•Load the scenarios.json file.

	•Prepare the dataset by extracting features like task points, task times, and total scenario points.

	•Train a TensorFlow neural network on the extracted features.

	•Save the trained model as a TensorFlow Lite model (task_model_optimized.tflite).

```bash
python trainMain.py
```

**Real Time Integration:**

Once the model is trained and saved as task_model_optimized.tflite, you can load it on a device that supports TensorFlow Lite for real-time predictions. This will allow you to use the model to predict the best sequence of tasks during a robotics competition.


Once the model is trained and saved as task_model_optimized.tflite, you can load it on a device that supports TensorFlow Lite for real-time predictions. This will allow you to use the model to predict the best sequence of tasks during a robotics competition.





# Java Native Build

## Overview
This Java-native version of the project aims to implement a scenario-based autonomous system prediction model without using Python or TensorFlow directly, instead it trains the model directly in Java. It does this simmilarly to the Python build, the Java versino is isnt complietly built out, and is much more error prone then the Python build. The Java application includes classes for defining tasks and scenarios, which can be used to optimize decision-making.

## Project Structure

- **`model.java`**: Defines the structure of the prediction model, including the logic for optimizing tasks based on points and time.
- **`main.java`**: The main entry point for the Java application. It initiates scenarios, runs predictions, and outputs results.
- **`Task.java`**: Defines the structure for individual tasks. Each task has a `task_name`, `points`, and `time_required`.
- **`scenario.java`**: Defines the structure for scenarios, which consist of multiple tasks and track the total points and time spent.
- **`helper.java`**: A utility class that contains helper methods for operations like loading tasks or scenarios from external sources.

## Requirements

- Java 8 or higher
- Maven (optional, for build management)

You can install Java and Maven as follows:

### For Ubuntu/Debian-based systems:
```bash
sudo apt install openjdk-11-jdk
sudo apt install maven


```
# General Important Notes
Note that whichever model you implement, the model will need to be saved as a pre-trained TF Lite file. The TF Lite file will need a helper script, written in Java, on board the bot, to facilitate real-time executable task updating.


**Required**
- The helper script needs to be able to input the most recent completed tasks data in the form of a scenario data package (coherent to that of which model you're using).
- A script with your bot's possible moves, mapped to possible tasks (matching your training data), and a method to concatenate them together to form strings of executable actions.

**A Note on Real-Time Integration:**

The bot will need to be able to receive the executable task strings from the helper script. However, there will also be logic needed to redirect the bot back to its defined list of executable actions. In other words, there needs to be robust edge case handling.
