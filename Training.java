import ai.djl.Model;
import ai.djl.basicdataset.ImageFolder;
import ai.djl.examples.training.util.AbstractTraining;
import ai.djl.examples.training.util.Arguments;
import ai.djl.examples.training.util.TrainingUtils;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.SimpleRepository;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.metrics.Accuracy;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import ai.djl.training.optimizer.learningrate.MultiFactorTracker;
import ai.djl.translate.Pipeline;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;

/*
 In training, multiple passes (or epochs) are made over the training data trying to find patterns and trends in the data, which are then
 stored in the model. During the process, the model is evaluated for accuracy using the validation data. The model is updated with findings
 over each epoch, which improves the accuracy of the model.
 */
public final class Training extends AbstractTraining {
    public static void main(String[] args) {
        new Training().runExample(args);
    }

    @Override
    protected void train(Arguments arguments) throws IOException {
        //identify source of training data
        String trainingDatasetRoot = "src/test/resources/imagefolder/train";

        //identify source of validation data
        String validateDatasetRoot = "src/test/resources/imagefolder/validate";

        //the height and width for pre-processing of the image
        int newHeight = 100;
        int newWidth = 100;

        //the batch size for training
        //represents number of training samples processed before the model is updated
        int batchSize = 32;

        //the number of classification labels: boots, sandals, shoes, slippers
        int numOfOutput = 4;

        //the number of passes over the complete dataset
        int numEpoch = 10;

        //the location to save the model
        String modelParamsPath = "build/logs";

        //the name of the model
        String modelParamsName = "shoeclassifier";

        // create training ImageFolder dataset
        ImageFolder trainingDataset = new ImageFolder.Builder()
                .setRepository(new SimpleRepository(Paths.get(trainingDatasetRoot)))
                .optPipeline(
                        // create preprocess pipeline
                        new Pipeline()
                                .add(new Resize(newWidth, newHeight))
                                .add(new ToTensor()))
                .setRandomSampling(batchSize)
                .build();

        trainingDataset.prepare();

        //create validation ImageFolder dataset
        ImageFolder validateDataset = new ImageFolder.Builder()
                .setRepository(new SimpleRepository(Paths.get(validateDatasetRoot)))
                .optPipeline(
                        // create preprocess pipeline
                        new Pipeline()
                                .add(new Resize(newWidth, newHeight))
                                .add(new ToTensor()))
                .setRandomSampling(batchSize)
                .build();

        validateDataset.prepare();

        trainDataSize = (int) (trainingDataset.size() / batchSize);
        validateDataSize = (int) (validateDataset.size() / batchSize);

        //set loss function
        //loss function evaluates model's predictions against the correct answer (during training)
        //higher numbers are bad - means model performed poorly; indicates more errors; want to minimize errors (loss)
        loss = Loss.softmaxCrossEntropyLoss();

        try (Model model = Models.getModel(numOfOutput, newHeight, newWidth)) {

            TrainingConfig config = setupTrainingConfig(batchSize, loss);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(metrics);
                trainer.setTrainingListener(this);

                Shape inputShape = new Shape(1, 3, newHeight, newWidth);

                // initialize trainer with proper input shape
                trainer.initialize(inputShape);

                //find the patterns in data
                TrainingUtils.fit(trainer, numEpoch, trainingDataset, validateDataset, "build/logs/training");

                //set model properties
                model.setProperty("Epoch", String.valueOf(numEpoch));
                model.setProperty("Accuracy", String.format("%.2f", getValidationAccuracy()));

                // save the model after done training for inference later
                //model saved as shoeclassifier-0000.params
                model.save(Paths.get(modelParamsPath), modelParamsName);
            }
        }
    }

    private static TrainingConfig setupTrainingConfig(int batchSize, Loss loss) {
        // epoch number to change learning rate
        int[] epoch = {3, 5, 8};
        int[] steps = Arrays.stream(epoch).map(k -> k * 60000 / batchSize).toArray();

        //initialize neural network weights using Xavier initializer
        Initializer initializer =
                new XavierInitializer(
                        XavierInitializer.RandomType.UNIFORM, XavierInitializer.FactorType.AVG, 2);

        //set the learning rate
        //adjusts weights of network based on loss
        MultiFactorTracker learningRateTracker =
                LearningRateTracker.multiFactorTracker()
                        .setSteps(steps)
                        .optBaseLearningRate(0.01f)
                        .optFactor(0.1f)
                        .optWarmUpBeginLearningRate(1e-3f)
                        .optWarmUpSteps(500)
                        .build();

        //set optimization technique
        //minimizes loss to produce better and faster results
        //Stochastic gradient descent
        Optimizer optimizer =
                Optimizer.sgd()
                        .setRescaleGrad(1.0f / batchSize)
                        .setLearningRateTracker(learningRateTracker)
                        .optMomentum(0.9f)
                        .optWeightDecays(0.001f)
                        .optClipGrad(1f)
                        .build();

        return new DefaultTrainingConfig(initializer, loss)
                .setOptimizer(optimizer)
                .addTrainingMetric(new Accuracy())
                .setBatchSize(batchSize);
    }
}
