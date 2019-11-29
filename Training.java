import ai.djl.Model;
import ai.djl.basicdataset.ImageFolder;
import ai.djl.examples.training.util.AbstractTraining;
import ai.djl.examples.training.util.Arguments;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.SimpleRepository;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.metrics.Accuracy;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import ai.djl.training.optimizer.learningrate.MultiFactorTracker;
import ai.djl.translate.Pipeline;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;


/*
 In training, multiple passes (or epochs) are made over the training data trying to find patterns and trends in the
 data, which are then stored in the model. During the process, the model is evaluated for accuracy using the
 validation data. The model is updated with findings over each epoch, which improves the accuracy of the model.
 */
public final class Training extends AbstractTraining {
    private static final Logger logger = LoggerFactory.getLogger(Training.class);

    //the number of classification labels: boots, sandals, shoes, slippers
    private static final int NUM_OF_OUTPUT = 4;

    //the height and width for pre-processing of the image
    private static final int NEW_HEIGHT = 100;
    private static final int NEW_WIDTH = 100;

    //represents number of training samples processed before the model is updated
    private static final int BATCH_SIZE = 32;

    //the number of passes over the complete dataset
    private static final int EPOCHS = 10;

    public static void main(String[] args) {
        new Training().runExample(args);
    }

    @Override
    protected void train(Arguments arguments) throws IOException {
        //identify source of training data
        String trainingDatasetRoot = "src/test/resources/imagefolder/train";

        //identify source of validation data
        String validateDatasetRoot = "src/test/resources/imagefolder/validate";

        //the location to save the model
        String modelParamsPath = "build/logs";

        //the name of the model
        String modelParamsName = "shoeclassifier";

        //create training ImageFolder dataset
        ImageFolder trainingDataset = initDataset(trainingDatasetRoot);

        //create validation ImageFolder dataset
        ImageFolder validateDataset = initDataset(validateDatasetRoot);

        batchSize = BATCH_SIZE; // there is a batchSize field in AbstractTraining

        //tell the machine how batches to process
        trainDataSize = (int) (trainingDataset.size() / BATCH_SIZE);
        validateDataSize = (int) (validateDataset.size() / BATCH_SIZE);

        //set loss function, which seeks to minimize errors
        //loss function evaluates model's predictions against the correct answer (during training)
        //higher numbers are bad - means model performed poorly; indicates more errors; want to minimize errors (loss)
        loss = Loss.softmaxCrossEntropyLoss();

        try (Model model = Models.getModel(NUM_OF_OUTPUT, NEW_HEIGHT, NEW_WIDTH)) { //empty model instance to hold patterns
            //setting training parameters (ie hyperparameters)
            TrainingConfig config = setupTrainingConfig(loss);

            try (Trainer trainer = model.newTrainer(config)) {
                //metrics collect and report key performance indicators, like accuracy
                trainer.setMetrics(metrics);
                trainer.setTrainingListener(this);

                Shape inputShape = new Shape(1, 3, NEW_HEIGHT, NEW_WIDTH);

                // initialize trainer with proper input shape
                trainer.initialize(inputShape);

                //find the patterns in data
                fit(trainer, trainingDataset, validateDataset, "build/logs/training");

                //set model properties
                model.setProperty("Epoch", String.valueOf(EPOCHS));
                model.setProperty("Accuracy", String.format("%.2f", getValidationAccuracy()));

                //save the model after done training for inference later
                //model saved as shoeclassifier-0000.params
                model.save(Paths.get(modelParamsPath), modelParamsName);
            }
        }
    }

    private ImageFolder initDataset(String datasetRoot) throws IOException {
        ImageFolder dataset = new ImageFolder.Builder()
                //retrieve the data
                .setRepository(new SimpleRepository(Paths.get(datasetRoot)))
                .optPipeline(
                        //create preprocess pipeline
                        new Pipeline()
                                .add(new Resize(NEW_WIDTH, NEW_HEIGHT))
                                .add(new ToTensor()))
                //random sampling; don't process the data in order
                .setSampling(BATCH_SIZE,true)
                .build();

        dataset.prepare();
        return dataset;
    }

    private static TrainingConfig setupTrainingConfig(Loss loss) {
        //epoch number to change learning rate
        int[] epoch = {3, 5, 8};
        int[] steps = Arrays.stream(epoch).map(k -> k * 60000 / BATCH_SIZE).toArray();

        //initialize neural network weights using Xavier initializer
        //weights dictate the importance of the input value
        //weights are random at first, then changed after each iteration to correct errors (uses learning rate)
        Initializer initializer =
                new XavierInitializer(
                        XavierInitializer.RandomType.UNIFORM, XavierInitializer.FactorType.AVG, 2);

        //set the learning rate
        //the amount the weights are adjusted based on loss (ie errors)
        //dictates how much to change the model in response to errors
        //sometimes called the step size
        MultiFactorTracker learningRateTracker =
                LearningRateTracker.multiFactorTracker()
                        .setSteps(steps)
                        .optBaseLearningRate(0.01f)
                        .optFactor(0.1f)
                        .optWarmUpBeginLearningRate(1e-3f)
                        .optWarmUpSteps(500)
                        .build();

        //set optimization technique, Stochastic Gradient Descent (SGD)
        //makes small adjustments to the network configuration to decrease errors
        //minimizes loss (i.e. errors) to produce better and faster results
        Optimizer optimizer =
                Optimizer.sgd()
                        .setRescaleGrad(1.0f / BATCH_SIZE)
                        .setLearningRateTracker(learningRateTracker)
                        .optMomentum(0.9f)
                        .optWeightDecays(0.001f)
                        .optClipGrad(1f)
                        .build();

        return new DefaultTrainingConfig(initializer, loss)
                .setOptimizer(optimizer)
                .addTrainingMetric(new Accuracy())
                .setBatchSize(BATCH_SIZE);
    }

    public void fit(Trainer trainer, Dataset trainingDataset, Dataset validateDataset,
                    String outputDir) throws IOException {

        //iterate over training dataset and produce a model
        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            //iterate over batches
            for (Batch batch : trainer.iterateDataset(trainingDataset)) {
                trainer.trainBatch(batch);
                trainer.step();
                batch.close();
            }

            //iterate over the validation dataset
            if (validateDataset != null) {
                for (Batch batch : trainer.iterateDataset(validateDataset)) {
                    trainer.validateBatch(batch);
                    batch.close();
                }
            }

            //reset training and validation metric at end of epoch
            trainer.resetTrainingMetrics();

            // save model at end of each epoch
            if (outputDir != null) {
                Model model = trainer.getModel();
                model.setProperty("Epoch", String.valueOf(epoch));
                model.save(Paths.get(outputDir), "resnetv1");
            }
        }
    }

}
