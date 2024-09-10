import uuid

class Task:
    def __init__(self, task_name, points, time_required, task_difficulty):
        """
        Task represents an individual task in a scenario.
        :param task_name: Name of the task
        :param points: Points the task is worth
        :param time_required: Time required to complete the task
        :param task_difficulty: Difficulty of the task
        """
        self.task_name = task_name
        self.points = points
        self.time_required = time_required
        self.task_difficulty = task_difficulty


class Scenario:
    def __init__(self, task_type):
        """
        Scenario contains a dynamic list of tasks, points, and time tracking.
        Each scenario is assigned a unique random tag (UUID).
        :param task_type: Type of the scenario (e.g., Autonomous Mode, Teleop Mode)
        """
        self.scenario_tag = str(uuid.uuid4())
        self.task_type = task_type
        self.tasks = []
        self.total_points = 0
        self.total_time_spent = 0
        self.total_task_difficulty = 0

    def add_task(self, task):
        """
        Add a task to the scenario and update the total points and time spent.
        :param task: Task object
        """
        self.tasks.append(task)
        self.total_points += task.points
        self.total_time_spent += task.time_required
        self.total_task_difficulty += task.task_difficulty

    def remove_task(self, task_name):
        """
        Remove a task by name and adjust total points and time.
        :param task_name: The name of the task to remove
        """
        for task in self.tasks:
            if task.task_name == task_name:
                self.total_points -= task.points
                self.total_time_spent -= task.time_required
                self.total_task_difficulty -= task.task_difficulty
                self.tasks.remove(task)
                break

    def get_tasks(self):
        """
        Get a summary of all tasks in the scenario.
        :return: A list of dictionaries representing each task
        """
        return [{"task_name": task.task_name, "points": task.points, "time_required": task.time_required, "task_difficulty": task.task_difficulty} for task in self.tasks]

    def get_scenario_summary(self):
        """
        Get a summary of the scenario.
        The task type is not returned since it is implied by the tasks performed.
        :return: A dictionary containing the scenario UUID, total points, total time spent, and tasks performed
        """
        return {
            "scenarioTag": self.scenario_tag,
            "totalPoints": self.total_points,
            "totalTimeSpent": self.total_time_spent,
            "totalTaskDifficulty": self.total_task_difficulty,
            "tasks": self.get_tasks()
        }


class ScenarioManager:
    def __init__(self):
        """
        ScenarioManager manages multiple scenarios dynamically.
        """
        self.scenarios = []

    def add_scenario(self, scenario):
        """
        Add a new scenario to the scenario manager.
        :param scenario: Scenario object
        """
        self.scenarios.append(scenario)

    def get_all_scenarios(self):
        """
        Get a summary of all scenarios.
        :return: A list of scenario summaries
        """
        return [scenario.get_scenario_summary() for scenario in self.scenarios]