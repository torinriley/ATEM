import json
import datetime
from scenario_classes import Task, Scenario, ScenarioManager

def main():
    scenario_manager = ScenarioManager()



    task1 = Task("Parked in Observation Zone", 3, 5, 5)
    task2 = Task("Level 1 Ascent", 2, 5, 5)
    task3 = Task("Sample in Net Zone", 2, 8, 4)
    task4 = Task("Sample in Low Basket", 4, 10, 3)

    scenario1 = Scenario("Autonomous Mode")
    scenario1.add_task(task1)
    scenario1.add_task(task2)
    scenario1.add_task(task3)
    scenario1.add_task(task4)

    scenario_manager.add_scenario(scenario1)

    all_scenarios = scenario_manager.get_all_scenarios()

    formatted_scenarios = []
    for i, scenario in enumerate(all_scenarios, 1):
        formatted_scenario = {
            "scenarioNumber": i,
            "scenarioTag": scenario['scenarioTag'],
            "totalPoints": scenario['totalPoints'],
            "totalTimeSpent": scenario['totalTimeSpent'],
            "totalTaskDifficulty": scenario['totalTaskDifficulty'],
            "tasksPerformed": scenario['tasks']
        }
        formatted_scenarios.append(formatted_scenario)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"scenarios_{timestamp}.json"

    with open(file_name, 'w') as f:
        json.dump(formatted_scenarios, f, indent=4)

    print(f"Realistic Auto Phase scenario output saved to {file_name}")

    for scenario in formatted_scenarios:
        print(f"Scenario {scenario['scenarioNumber']} (UUID: {scenario['scenarioTag']}):")
        print(f"  Total Points: {scenario['totalPoints']}")
        print(f"  Total Time Spent: {scenario['totalTimeSpent']} seconds")
        print(f"  Total Task Difficulty: {scenario['totalTaskDifficulty']}")
        print("  Tasks Performed:")
        for task in scenario['tasksPerformed']:
            print(f"    - {task['task_name']} (Points: {task['points']}, Time: {task['time_required']} seconds, Difficulty: {task['task_difficulty']})")
        print("-" * 50)

if __name__ == "__main__":
    main()