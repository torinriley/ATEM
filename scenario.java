import java.util.ArrayList;
import java.util.HashMap;

public class scenario {
    String taskType;
    int PointsPerTask;
    int TleftTask;
    int NumberTasksCompleted;
    int totalPoints;

    static ArrayList<scenario> scenarios = new ArrayList<>();

    public scenario(String taskType, int PointsPerTask, int TleftTask, int NumberTasksCompleted, int totalPoints) {
        this.taskType = taskType;
        this.PointsPerTask = PointsPerTask;
        this.TleftTask = TleftTask;
        this.NumberTasksCompleted = NumberTasksCompleted;
        this.totalPoints = totalPoints;
    }

    public void addScenario() {
        scenarios.add(this);
    }

    public HashMap<scenario, HashMap<String, Object>> viewScenario(tasks taskInstance) {
        HashMap<scenario, HashMap<String, Object>> scenarioDetails = new HashMap<>();

        for (scenario instance : scenarios) {
            String mostRecentKey = taskInstance.getMostRecentKey();
            ArrayList<String> tasksList = taskInstance.getTasksList(mostRecentKey);
            int pointsPerTask = taskInstance.getPointsPerTask(mostRecentKey);
            int timeLeft = taskInstance.getTimeLeft(mostRecentKey);
            int numberOfTasksCompleted = tasksList.size();
            int totalPoints = numberOfTasksCompleted * pointsPerTask;

            HashMap<String, Object> details = new HashMap<>();
            details.put("tasksList", tasksList);
            details.put("pointsPerTask", pointsPerTask);
            details.put("timeLeft", timeLeft);
            details.put("numberOfTasksCompleted", numberOfTasksCompleted);
            details.put("totalPoints", totalPoints);

            scenarioDetails.put(instance, details);
        }

        return scenarioDetails;
    }
}