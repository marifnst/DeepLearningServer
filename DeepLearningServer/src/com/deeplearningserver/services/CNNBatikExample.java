package com.deeplearningserver.services;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.deeplearningserver.dependency.batik.BatikDataSetIterator;
import com.deeplearningserver.util.Variables;

/**
 * Created by willow on 5/11/15.
 */
public class CNNBatikExample {

    private static final Logger log = LoggerFactory.getLogger(CNNBatikExample.class);

    public static void main(String[] args) throws Exception {
        int nChannels = 1;
        int outputNum = 27;
        int numSamples = 60;
        int batchSize = 10;
        int iterations = 1;
        int splitTrainNum = (int) (batchSize * .8);
        int seed = 123;
        int listenerFreq = iterations/5;
        
        System.out.println(new Random(seed).nextInt());
        
        DataSet batik;
        SplitTestAndTrain trainTest;
        DataSet trainInput = null;
        List<INDArray> testInput = new ArrayList<>();
        List<INDArray> testLabels = new ArrayList<>();

        log.info("Load data....");
        DataSetIterator batikIter = new BatikDataSetIterator(batchSize, numSamples, false);
        
        log.info("Build model....");
		MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(iterations)
				.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.list(3)
				.layer(0,new ConvolutionLayer.Builder(10, 10).stride(2, 2)
								.nIn(nChannels).nOut(6)
								.weightInit(WeightInit.XAVIER)
								.activation("relu").build())
				.layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2, 2 }).build())
				.layer(2, new OutputLayer.Builder(
								LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
								.nOut(outputNum)
								.weightInit(WeightInit.XAVIER)
								.activation("softmax")
								.build()).backprop(true)
				.pretrain(false);

        new ConvolutionLayerSetup(builder, Variables.BASE_DATA_HEIGHT, Variables.BASE_DATA_WIDTH, nChannels);

        MultiLayerConfiguration conf = builder.build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info("Train model....");
        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));
        while(batikIter.hasNext()) {
        	batik = batikIter.next();
            trainTest = batik.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
            trainInput = trainTest.getTrain(); // get feature matrix and labels for training
//            System.out.println(trainInput.getFeatureMatrix().rows());
            testInput.add(trainTest.getTest().getFeatureMatrix());
            testLabels.add(trainTest.getTest().getLabels());
            model.fit(trainInput);
        }

        log.info("Evaluate weights....");

        log.info("Evaluate model....");
//        System.out.println(trainInput);
//        System.out.println(testInput.size());        
        Evaluation eval = new Evaluation(outputNum);
        for(int i = 0; i < testInput.size(); i++) {
            INDArray output = model.output(testInput.get(i));
            eval.eval(testLabels.get(i), output);
        }
        
        INDArray output = model.output(testInput.get(0));
        eval.eval(testLabels.get(0), output);
        log.info(eval.stats());
        log.info("****************Example finished********************");


    }
}
