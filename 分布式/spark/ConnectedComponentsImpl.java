package DSPPCode.spark.connected_components.impl;

import DSPPCode.spark.connected_components.question.ConnectedComponents;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ConnectedComponentsImpl extends ConnectedComponents {

  @Override
  public JavaPairRDD<String, Integer> getcc(JavaRDD<String> text) {
    // Step 1: Parse input data into (vertex, [neighbors]) pairs
    JavaPairRDD<String, List<String>> neighbors = text.mapToPair(line -> {
      String[] parts = line.split("\t");
      String vertex = parts[0];
      List<String> neighborsList = new ArrayList<>(Arrays.asList(parts).subList(1, parts.length));
      return new Tuple2<>(vertex, neighborsList);
    });

    // Step 2: Initialize each vertex to be in its own component
    JavaPairRDD<String, Integer> components = neighbors.mapToPair(vertex ->
        new Tuple2<>(vertex._1, Integer.parseInt(vertex._1))
    );

    boolean hasChanged;
    do {
      // Step 3: Propagate the component ID to each neighbor
      JavaPairRDD<String, Integer> propagatedComponents = neighbors.join(components).flatMapToPair(joined -> {
        String vertex = joined._1;
        Integer component = joined._2._2;
        List<Tuple2<String, Integer>> results = new ArrayList<>();
        results.add(new Tuple2<>(vertex, component));
        for (String neighbor : joined._2._1) {
          results.add(new Tuple2<>(neighbor, component));
        }
        return results.iterator();
      });

      // Step 4: Take the minimum component ID for each vertex
      JavaPairRDD<String, Integer> newComponents = propagatedComponents.reduceByKey(Math::min);

      // Step 5: Check if components have changed
      hasChanged = isChange(components, newComponents);

      // Update components for the next iteration
      components = newComponents;

    } while (hasChanged);

    // Step 6: Return the final components RDD
    return components;
  }

}
