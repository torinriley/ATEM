import java.util.HashMap;

public class CurrentScenario {
    String lastTaskCompleted;
    int pointsFromLastTask;
    int timeLeft;
    int opponentAllianceScore;
    String nextPlannedTask;

    static CurrentScenario currentScenario;

    public CurrentScenario(String lastTaskCompleted, int pointsFromLastTask, int timeLeft, int opponentAllianceScore, String nextPlannedTask) {
        this.lastTaskCompleted = lastTaskCompleted;
        this.pointsFromLastTask = pointsFromLastTask;
        this.timeLeft = timeLeft;
        this.opponentAllianceScore = opponentAllianceScore;
        this.nextPlannedTask = nextPlannedTask;
    }

    public static void updateScenario(String lastTaskCompleted, int pointsFromLastTask, int timeLeft, int opponentAllianceScore, String nextPlannedTask) {
        currentScenario = new CurrentScenario(lastTaskCompleted, pointsFromLastTask, timeLeft, opponentAllianceScore, nextPlannedTask);
    }

    public static HashMap<String, Object> getCurrentScenario() {
        HashMap<String, Object> scenarioDetails = new HashMap<>();
        scenarioDetails.put("lastTaskCompleted", currentScenario.lastTaskCompleted);
        scenarioDetails.put("pointsFromLastTask", currentScenario.pointsFromLastTask);
        scenarioDetails.put("timeLeft", currentScenario.timeLeft);
        scenarioDetails.put("opponentAllianceScore", currentScenario.opponentAllianceScore);
        scenarioDetails.put("nextPlannedTask", currentScenario.nextPlannedTask);
        return scenarioDetails;
    }
}