package DSPPCode.spark.perceptron.impl;

import DSPPCode.spark.perceptron.question.DataPoint;
import DSPPCode.spark.perceptron.question.IterationStep;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.broadcast.Broadcast;

public class IterationStepImpl extends IterationStep {

  @Override
  public Broadcast<double[]> createBroadcastVariable(JavaSparkContext sc, double[] localVariable) {
    return sc.broadcast(localVariable);
  }

  @Override
  public boolean termination(double[] old, double[] newWeightsAndBias) {
    double sum = 0.0;
    for (int i = 0; i < old.length; i++) {
      sum += (newWeightsAndBias[i] - old[i]) * (newWeightsAndBias[i] - old[i]);
    }
    return sum < THRESHOLD;
  }

  @Override
  public double[] runStep(JavaRDD<DataPoint> points, Broadcast<double[]> broadcastWeightsAndBias) {
    double[] weightsAndBias = broadcastWeightsAndBias.value();

    JavaRDD<double[]> gradients = points.map(new ComputeGradient(weightsAndBias));
    double[] gradientSum = gradients.reduce(new VectorSum());

    double[] newWeightsAndBias = new double[weightsAndBias.length];
    for (int i = 0; i < weightsAndBias.length; i++) {
      newWeightsAndBias[i] = weightsAndBias[i] + STEP * gradientSum[i];
    }

    return newWeightsAndBias;
  }

  public static class VectorSum extends IterationStep.VectorSum {
    @Override
    public double[] call(double[] a, double[] b) throws Exception {
      double[] result = new double[a.length];
      for (int i = 0; i < a.length; i++) {
        result[i] = a[i] + b[i];
      }
      return result;
    }
  }

  public static class ComputeGradient extends IterationStep.ComputeGradient {
    public ComputeGradient(double[] weightsAndBias) {
      super(weightsAndBias);
    }

    @Override
    public double[] call(DataPoint dataPoint) throws Exception {
      double[] gradient = new double[weightsAndBias.length];
      double y = dataPoint.y;
      double[] x = dataPoint.x;

      double dotProduct = 0.0;
      for (int i = 0; i < x.length; i++) {
        dotProduct += weightsAndBias[i] * x[i];
      }
      dotProduct += weightsAndBias[weightsAndBias.length - 1]; // Add the bias term

      if (y * dotProduct <= 0) { // Misclassified point or on decision boundary
        for (int i = 0; i < x.length; i++) {
          gradient[i] = y * x[i];
        }
        gradient[gradient.length - 1] = y; // Bias term
      }

      return gradient;
    }
  }
}
