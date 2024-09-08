package data;
import java.util.ArrayList;
import java.util.HashMap;

public class Task {
    private HashMap<String, ArrayList<String>> taskMap = new HashMap<>();
    private HashMap<String, Integer> pointsMap = new HashMap<>();
    private HashMap<String, Integer> timeLeftMap = new HashMap<>();

    // Add a task to the map
    public void addTask(String key, ArrayList<String> value, int pointsPerTask, int timeLeft) {
        taskMap.put(key, value);
        pointsMap.put(key, pointsPerTask);
        timeLeftMap.put(key, timeLeft);
    }

    // Get the most recent key
    public String getMostRecentKey() {
        String mostRecentKey = null;
        for (String key : taskMap.keySet()) {
            mostRecentKey = key;
        }
        return mostRecentKey;
    }

    // Get the tasks list for a given key
    public ArrayList<String> getTasksList(String key) {
        return taskMap.get(key);
    }

    // Get the points per task for a given key
    public int getPointsPerTask(String key) {
        return pointsMap.get(key);
    }

    // Get the time left for a given key
    public int getTimeLeft(String key) {
        return timeLeftMap.get(key);
    }
}
