import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/*
   Uses the model to generate a prediction called an inference
 */
public class Inference {
    private static final Logger logger = LoggerFactory.getLogger(Inference.class);

    //values must match settings used during training
    //the number of classification labels: boots, sandals, shoes, slippers
    private static final int NUM_OF_OUTPUT = 4;

    //the height and width for pre-processing of the image
    private static final int NEW_HEIGHT = 100;
    private static final int NEW_WIDTH = 100;

    public static void main(String[] args) throws ModelException, TranslateException, IOException {
        Classifications classifications = new Inference().predict();
        logger.info("{}", classifications);
    }

    private Classifications predict() throws IOException, ModelException, TranslateException  {
        //the location to the model saved during training
        String modelParamsPath = "build/logs";

        //the name of the model set during training
        String modelParamsName = "shoeclassifier";

        // the path of image to classify
        // 0-boots; 1-sandals; 2-shoes; 3-slippers

        String imageFilePath = "src/test/resources/sandals.jpg";
        //String imageFilePath = "src/test/resources/boots.jpg";
        //String imageFilePath = "src/test/resources/shoes.jpg";
        //String imageFilePath = "src/test/resources/slippers.jpg";

        //Load the image file from the path
        BufferedImage img = BufferedImageUtils.fromFile(Paths.get(imageFilePath));

        //holds the probability score per label
        Classifications predictResult;

        try (Model model = Models.getModel(NUM_OF_OUTPUT, NEW_HEIGHT, NEW_WIDTH)) { //empty model instance
            //load the model
            model.load(Paths.get(modelParamsPath), modelParamsName);

            //define a translator for pre and post processing
            //out of the box this translator converts images to ResNet friendly ResNet 18 shape
            Translator<BufferedImage, Classifications> translator = new MyTranslator();

            //run the inference using a Predictor
            try (Predictor<BufferedImage, Classifications> predictor = model.newPredictor(translator)) {
                predictResult = predictor.predict(img);
            }
        }

        return predictResult;
    }

    private static final class MyTranslator implements Translator<BufferedImage, Classifications> {

        private List<String> classes;

        MyTranslator() {
            classes = IntStream.range(0, NUM_OF_OUTPUT).mapToObj(String::valueOf).collect(Collectors.toList());
        }

        @Override
        public NDList processInput(TranslatorContext ctx, BufferedImage input) {
            NDArray array = BufferedImageUtils.toNDArray(ctx.getNDManager(), input, NDImageUtils.Flag.COLOR);
            Pipeline pipeline = new Pipeline()
                    .add(new Resize(NEW_WIDTH, NEW_HEIGHT))
                    .add(new ToTensor());
            return pipeline.transform(new NDList(array));
        }

        @Override
        public Classifications processOutput(TranslatorContext ctx, NDList list) {
            NDArray probabilities = list.singletonOrThrow().softmax(0);
            return new Classifications(classes, probabilities);
        }
    }
}
