import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.zoo.cv.classification.ResNetV1;

/*
    Use a neural network (ResNet-50) to train the model
    ResNet-50 is a deep residual network with 50 layers; good for image classification
 */
public class Models {
    public static ai.djl.Model getModel(int numOfOutput, int height, int width) {
        //create new instance of an empty model
        ai.djl.Model model = ai.djl.Model.newInstance();

        //Block is composable unit that forms a neural network; combine them like Lego blocks to form a complex network
        Block resNet50 =
                //construct the network
                new ResNetV1.Builder()
                        .setImageShape(new Shape(3, height, width))
                        .setNumLayers(50)
                        .setOutSize(numOfOutput)
                        .build();

        //set the neural network to the model
        model.setBlock(resNet50);
        return model;
    }
}
