

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
public class WekaClassification {

public void Predict(Instance sample) throws Exception {
 
		double class1 = model.classifyInstance(sample);
 
		System.out.println("first: " + class1);
}	
	
	
public static BufferedReader readDataFile(String filename) {
BufferedReader inputReader = null;
try {
	inputReader = new BufferedReader(new FileReader(filename));
} catch (FileNotFoundException ex) {
	System.err.println("File not found: " + filename);
}
	return inputReader;
}

public static Evaluation classify(Classifier model,
Instances trainingSet, Instances testingSet) throws Exception {
	Evaluation evaluation = new Evaluation(trainingSet);
	model.buildClassifier(trainingSet);
	evaluation.evaluateModel(model, testingSet);
	return evaluation;
}
public static double calculateAccuracy(FastVector predictions) {
double correct = 0;
for (int i = 0; i < predictions.size(); i++) {
	NominalPrediction np = (NominalPrediction) predictions.elementAt(i);
if (np.predicted() == np.actual()) {
correct++;
}
}
	return 100 * correct / predictions.size();
}

public static Instances[][] crossValidationSplit(Instances data, int numberOfFolds) {
	Instances[][] split = new Instances[2][numberOfFolds];
for (int i = 0; i < numberOfFolds; i++) {
	split[0][i] = data.trainCV(numberOfFolds, i);
	split[1][i] = data.testCV(numberOfFolds, i);
}
return split;
}

static Classifier model;


public WekaClassification(String dataset_path) throws Exception {
	
	CSVLoader loader = new CSVLoader();
	loader.setFieldSeparator(",");
	loader.setSource(new File(dataset_path));
	Instances data = loader.getDataSet();	

	//zadnji atribut je klasa
	data.setClassIndex(data.numAttributes() - 1);

	
	//Cross validacija u 10 koraka
	Instances[][] split = crossValidationSplit(data, 10);
	
	// Podeliti skup u trening i test
	Instances[] trainingSplits = split[0];
	Instances[] testingSplits = split[1];
	
	// Kreirati klasifikator
	try {
		model = new IBk();
	} catch (Exception e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}


	FastVector predictions = new FastVector();
for (int i = 0; i < trainingSplits.length; i++) {
	Evaluation validation = classify(model, trainingSplits[i], testingSplits[i]);
	predictions.appendElements(validation.predictions());

}
// Izracunati performanse algoritma
	double accuracy = calculateAccuracy(predictions);
	System.out.println("Accuracy of " + model.getClass().getSimpleName() + ": "
			+ String.format("%.2f%%", accuracy)+ "\n---------------------------------");
}


public static void main(String[] args) throws Exception {
{
	try {
		
		long start = System.currentTimeMillis();

	
	WekaClassification predictor=new WekaClassification("stability.csv");

		// ...
		long finish = System.currentTimeMillis();
		long timeElapsed = finish - start;
		System.out.println(timeElapsed);
	}
	catch (Exception e)
	{
		System.out.println(e.getMessage());
	}
	}
}

}
