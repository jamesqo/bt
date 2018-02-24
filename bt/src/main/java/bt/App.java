package bt;

import java.io.IOException;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.data.manipulator.LexicalStructureElementManipulator;
import it.uniroma2.sag.kelp.data.representation.structure.similarity.LexicalStructureElementSimilarity;
import it.uniroma2.sag.kelp.data.representation.structure.similarity.compositional.sum.CompositionalNodeSimilaritySum;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.cache.FixIndexSquaredNormCache;
import it.uniroma2.sag.kelp.kernel.cache.FixSizeKernelCache;
import it.uniroma2.sag.kelp.kernel.standard.NormalizationKernel;
import it.uniroma2.sag.kelp.kernel.tree.PartialTreeKernel;
import it.uniroma2.sag.kelp.kernel.tree.SmoothedPartialTreeKernel;
import it.uniroma2.sag.kelp.kernel.tree.SubSetTreeKernel;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.libsvm.BinaryCSvmClassification;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassClassificationEvaluator;
import it.uniroma2.sag.kelp.wordspace.Wordspace;

public class App {
    public static void main(String[] args) {
		try {
			/*
			 * Initializing the Log level
			 */
			System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "INFO");

			/*
			 * Read both training and testing dataset
			 */
			SimpleDataset trainingSet = new SimpleDataset();
			trainingSet.populate("src/main/resources/qc/train_5500.coarse.klp.gz");
			SimpleDataset testSet = new SimpleDataset();
			testSet.populate("src/main/resources/qc/TREC_10.coarse.klp.gz");

			/*
			 * Select a specific kernel from the existing ones:
			 * - bow: a Linear Kernel applied to a boolean Bag-of-Word vector
			 * - stk: a Subset Tree Kernel 
			 * - ptk: a Partial Tree Kernel
			 * - sptk: a Smoothed Partial Tree Kernel
			 * - csptk: Compositionally Smoothed Partial Tree Kernel
			 */
			String kernelId = "bow";

			/*
			 * print some statistics
			 */
			System.out.println("Training set statistics");
			System.out.print("Examples number ");
			System.out.println(trainingSet.getNumberOfExamples());
			/*
			 * print the number of train and test example for each class
			 */
			for (Label l : trainingSet.getClassificationLabels()) {
				System.out.println("Training Label " + l.toString() + " " + trainingSet.getNumberOfPositiveExamples(l));
				System.out.println("Training Label " + l.toString() + " " + trainingSet.getNumberOfNegativeExamples(l));

				System.out.println("Test Label " + l.toString() + " " + testSet.getNumberOfPositiveExamples(l));
				System.out.println("Test Label " + l.toString() + " " + testSet.getNumberOfNegativeExamples(l));
			}

			/*
			 * Initialize the proper kernel function
			 */
			Kernel usedKernel = getQCKernelFunction(trainingSet, testSet, kernelId);

			JacksonSerializerWrapper serializer = new JacksonSerializerWrapper();

			/*
			 * Instantiate the SVM learning Algorithm. This is a binary
			 * classifier that will be transparently duplicated from the
			 * Multi-class classifier
			 */
			BinaryCSvmClassification svmSolver = new BinaryCSvmClassification();
			/*
			 * Set the kernel
			 */
			svmSolver.setKernel(usedKernel);
			/*
			 * Set the C parameter
			 */
			svmSolver.setCn(3);
			/*
			 * Enamble the fairness: in each binary classifier, the
			 * regularization parameter of the positive examples is multiplied
			 * of a coefficient that is
			 * number_of_negative_examples/number_of_positive_examples
			 */
			svmSolver.setFairness(true);
			/*
			 * Instantiate the multi-class classifier that apply a One-vs-All
			 * schema
			 */
			OneVsAllLearning ovaLearner = new OneVsAllLearning();
			/*
			 * Use the binary classifier defined above
			 */
			ovaLearner.setBaseAlgorithm(svmSolver);
			ovaLearner.setLabels(trainingSet.getClassificationLabels());
			/*
			 * The classifier can be serialized
			 */
			serializer.writeValueOnFile(ovaLearner,
					"src/main/resources/qc/learningAlgorithmSpecificationFromJavaCode.klp");
			/*
			 * Learn and get the prediction function
			 */
			ovaLearner.learn(trainingSet);
			Classifier f = ovaLearner.getPredictionFunction();
			/*
			 * Write the model (aka the Classifier for further use)
			 */
			serializer.writeValueOnFile(f, "src/main/resources/qc/classificationAlgorithm.klp");

			/*
			 * Classify examples and compute the accuracy, i.e. the percentage
			 * of questions that are correctly classified
			 */
			MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator(
					trainingSet.getClassificationLabels());
			for (Example e : testSet.getExamples()) {
				// Predict the class
				ClassificationOutput p = f.predict(testSet.getNextExample());
				evaluator.addCount(e, p);
				System.out.println("Question:\t" + e.getRepresentation("quest"));
				System.out.println("Original class:\t" + e.getClassificationLabels());
				System.out.println("Predicted class:\t" + p.getPredictedClasses());
				System.out.println();
			}

			System.out.println("Accuracy: " + evaluator.getAccuracy());
		} catch (Exception e1) {
			e1.printStackTrace();
		}
    }

	/**
	 * Get one of the kernel functions used in the Question Classification
	 * examples.
	 * 
	 * @param trainingSet
	 *            The training dataset (needed in some optimizations)
	 * @param testSet
	 *            The test dataset (needed in some optimizations)
	 * @param kernelId
	 *            Specifies the desired kernel. One of the following parameters
	 *            should be set:<br>
	 *            - bow: a Linear Kernel applied to a boolean Bag-of-Word
	 *            vector; <br>
	 *            - stk: a Subset Tree Kernel; <br>
	 *            - ptk: a Partial Tree Kernel; <br>
	 *            - sptk: a Smoothed Partial Tree Kernel; <br>
	 *            - csptk: Compositionally Smoothed Partial Tree Kernel. <br>
	 *            For more details about these kernel, see the class description.
	 * 			
	 * @return the kernel function
	 * @throws IOException
	 */
	public static Kernel getQCKernelFunction(SimpleDataset trainingSet, SimpleDataset testSet, String kernelId)
			throws IOException {

		Kernel usedKernel = null;
		/*
		 * Set the cache size
		 */
		int cacheSize = trainingSet.getNumberOfExamples() + testSet.getNumberOfExamples();
		/*
		 * Linear kernel over Bag-of-word representation. Expected accuracy:
		 * 86.2%
		 */
		if (kernelId.equalsIgnoreCase("bow")) {
			// The representation on which the kernel operates
			String vectorRepresentationName = "bow";
			// Definition of the linear kernel
			Kernel linearKernel = new LinearKernel(vectorRepresentationName);
			// This cache stores the norm of the kernel BEFORE normalizing.
			linearKernel.setSquaredNormCache(new FixIndexSquaredNormCache(cacheSize));
			// The kernel is normalized.
			Kernel normLinearKernel = new NormalizationKernel(linearKernel);
			usedKernel = normLinearKernel;
		} else
		/*
		 * Subset Tree Kernel over the Grammatical Relation Centered Tree
		 * representation. Expected accuracy: 91.4%
		 */
		if (kernelId.equalsIgnoreCase("stk")) {
			// The representation on which the kernel operates
			String treeRepresentationName = "grct";
			// Definition of the Subset Tree Kernel
			Kernel stkgrct = new SubSetTreeKernel(0.4f, treeRepresentationName);
			// This cache stores the norm of the kernel BEFORE normalizing.
			stkgrct.setSquaredNormCache(new FixIndexSquaredNormCache(cacheSize));
			// The kernel is normalized.
			Kernel normPtkGrct = new NormalizationKernel(stkgrct);
			usedKernel = normPtkGrct;
		} else
		/*
		 * Partial Tree Kernel over the Grammatical Relation Centered Tree
		 * representation. Expected accuracy: 91.0%
		 */
		if (kernelId.equalsIgnoreCase("ptk")) {
			// The representation on which the kernel operates
			String treeRepresentationName = "grct";
			// Kernel for the grct representation
			Kernel ptkgrct = new PartialTreeKernel(0.4f, 0.4f, 5f, treeRepresentationName);
			// This cache stores the norm of the kernel BEFORE normalizing.
			ptkgrct.setSquaredNormCache(new FixIndexSquaredNormCache(cacheSize));
			// The kernel is normalized.
			Kernel normPtkGrct = new NormalizationKernel(ptkgrct);
			usedKernel = normPtkGrct;
		} else
		/*
		 * Smoothed Partial Tree Kernel over the Lexically Centered Tree
		 * representation. Expected accuracy: 94.0%
		 */
		if (kernelId.equalsIgnoreCase("sptk")) {
			// The representation on which the kernel operates
			String treeRepresentationName = "lct";
			// The node similarity function between lexical nodes is based
			// on a Distributional Model, as discussed in
			// [Croce et al.(2011)]
			String matrixPath = "src/main/resources/wordspace/wordspace_qc.txt.gz";
			// The word space containing the vector representation of words
			// represented in lexical nodes is loaded
			Wordspace wordspace = new Wordspace(matrixPath);
			// This manipulator assigns vectors to lexical nodes. It allows
			// to speed-up computations: otherwise each time the similarity
			// between two nodes is evaluated, the corresponding vectors are
			// retrieved in the word space, with additional operational
			// costs.
			LexicalStructureElementManipulator lexManipulator = new LexicalStructureElementManipulator(wordspace,
					treeRepresentationName);
			trainingSet.manipulate(lexManipulator);
			testSet.manipulate(lexManipulator);
			// This class implements a similarity function between lexical
			// nodes based on the Word space
			LexicalStructureElementSimilarity similarityWordspace = new LexicalStructureElementSimilarity(wordspace);
			// The kernel operating over the lct representation
			Kernel sptklct = new SmoothedPartialTreeKernel(0.4f, 0.4f, 0.2f, 0.01f, similarityWordspace,
					treeRepresentationName);
			// This cache stores the norm of the kernel BEFORE normalizing.
			sptklct.setSquaredNormCache(new FixIndexSquaredNormCache(cacheSize));
			// The kernel is normalized.
			NormalizationKernel normalizedSptkLct = new NormalizationKernel(sptklct);
			usedKernel = normalizedSptkLct;
		} else
		/*
		 * Compositionally Smoothed Partial Tree Kernel over the Compositional
		 * Lexically Centered Tree representation. Expected accuracy: 95.0%
		 */if (kernelId.equalsIgnoreCase("csptk")) {
			// The representation on which the kernel operates
			String treeRepresentationName = "clct";
			// The node similarity function between lexical nodes is based
			// on a Distributional Model, as in [Annesi et al.(2014)]
			String matrixPath = "src/main/resources/wordspace/wordspace_qc.txt.gz";
			Wordspace wordspace = new Wordspace(matrixPath);
			// This manipulator assigns vectors to lexical nodes. It allows
			// to speed-up computations: otherwise each time the similarity
			// between two nodes is evaluated, the corresponding vectors are
			// retrieved in the word space, with additional operational
			// costs.
			LexicalStructureElementManipulator lexManipulator = new LexicalStructureElementManipulator(wordspace,
					treeRepresentationName);
			trainingSet.manipulate(lexManipulator);
			testSet.manipulate(lexManipulator);
			// Compositional nodes syntactic nodes are represented as vector
			// that is the sum of the vector representing the syntactic head
			// and modifier, as discussed in [Annesi et al(2014)]
			CompositionalNodeSimilaritySum compSS = new CompositionalNodeSimilaritySum();
			compSS.setWordspace(wordspace);
			compSS.setRepresentationToBeEnriched(treeRepresentationName);
			// This manipulator assigns vectors to "compositional syntactic"
			// nodes. It allows to speed-up computations: otherwise each
			// time the similarity between two nodes is evaluated, the
			// corresponding vectors are retrieved in the word space, with
			// additional operational costs.
			trainingSet.manipulate(compSS);
			testSet.manipulate(compSS);
			// The kernel operating over the clct representation
			Kernel sptkcgrct = new SmoothedPartialTreeKernel(0.4f, 0.4f, 1f, 0.01f, compSS, treeRepresentationName);
			// This cache stores the norm of the kernel BEFORE normalizing.
			sptkcgrct.setSquaredNormCache(new FixIndexSquaredNormCache(cacheSize));
			// The kernel is normalized.
			Kernel normSptklct = new NormalizationKernel(sptkcgrct);
			usedKernel = normSptklct;
		} else {
			System.err.println("The kernel " + kernelId + " has not been defined.");
		}
		/*
		 * Set cache to the kernel
		 */
		usedKernel.setKernelCache(new FixSizeKernelCache(cacheSize));
		return usedKernel;
	}
}
