import java.util.ArrayList;

public class model {
    priavate TensorFlowModel tfModel;

    public Model() {
        tfModel = new TensorFlowModel();
    }
    
}

public void trainModel(List<Scenario> scenarios) {
    float [] inputs  = createFeatures(scenarios);
    float [] outputs = createLabels(scenarios);


    this.tfModel.train(inputs, labels);
}

public List<Task> optimizeTasks (Scenario scenario) {
    float [] [] scnarioFeatures = createFeaturesForScenario(scenario);
    int [] optimizedTaskOrder = this.tfModel.predict(scenarioFeatures);
    return reorderTasks(optomizedTaskOrder, scenario);
}

private float[][]createFeatures(List<Scenario> scenarios) {
    return new float[scenarios.size()][];
}

private float[][] createLabels(List<Scenario> scenarios) {
    return new float[scenarios.size()][];
}


privtae float [] [] createFeaturesForScenario(Scenario scenarios) {
    return new float[][]{};
}

private List<Task> redorderTasks(int[] otimizedOrder, Scenario scenario) {
    return new ArrayList<Task>();
}







//INPUT
//ai model is trained on scenairos, (in other words each instance of the scenarios object using the viewScenario method form the scenarios class (src/data/scenario.java))
//takes in each create dinstance of the scnario object

//OPTOMIZATION//
//optomizes: the spesific tasks to complete and the order of these tasks to optomize the number of points
// MUST use ONLY tensorflow
// MUST save the model in a pre-trained format for later acsess and use
    

//OUTPUT//
//returns the optimized order of spesific tasks to complete
