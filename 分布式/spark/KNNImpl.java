package DSPPCode.spark.knn.impl;

import DSPPCode.spark.knn.question.Data;
import DSPPCode.spark.knn.question.KNN;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.PairFunction;
import scala.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

public class KNNImpl extends KNN {

  public KNNImpl(int k) {
    super(k);
  }

  @Override
  public JavaPairRDD<Data, Data> kNNJoin(JavaRDD<Data> trainData, JavaRDD<Data> queryData) {
    return queryData.cartesian(trainData);
  }

  @Override
  public JavaPairRDD<Integer, Tuple2<Integer, Double>> calculateDistance(JavaPairRDD<Data, Data> data) {
    return data.mapToPair(pair -> {
      Data query = pair._1;
      Data train = pair._2;
      double distance = 0.0;
      for (int i = 0; i < query.x.length; i++) {
        distance += Math.pow(query.x[i] - train.x[i], 2);
      }
      distance = Math.sqrt(distance);
      return new Tuple2<>(query.id, new Tuple2<>(train.y, distance));
    });
  }

  @Override
  public JavaPairRDD<Integer, Integer> classify(JavaPairRDD<Integer, Tuple2<Integer, Double>> data) {
    // 先在map阶段进行局部排序和聚合，减少shuffle的数据量
    return data.aggregateByKey(
        new ArrayList<Tuple2<Integer, Double>>(),
        (list, value) -> {
          list.add(value);
          return list;
        },
        (list1, list2) -> {
          list1.addAll(list2);
          return list1;
        }
    ).mapToPair(pair -> {
      int queryId = pair._1;
      List<Tuple2<Integer, Double>> neighborsList = pair._2;

      // 使用Stream进行排序
      List<Tuple2<Integer, Double>> sortedNeighborsList = neighborsList.stream()
          .sorted(Comparator.comparingDouble(Tuple2::_2))
          .limit(k)
          .collect(Collectors.toList());

      Map<Integer, Integer> labelCounts = new HashMap<>();

      for (Tuple2<Integer, Double> nearest : sortedNeighborsList) {
        labelCounts.put(nearest._1, labelCounts.getOrDefault(nearest._1, 0) + 1);
      }

      int majorityLabel = labelCounts.entrySet().stream().min((a, b) -> {
            if (!a.getValue().equals(b.getValue())) {
              return b.getValue() - a.getValue();
            } else {
              return a.getKey() - b.getKey();
            }
          })
          .get()
          .getKey();

      return new Tuple2<>(queryId, majorityLabel);
    });
  }
}
