package DSPPCode.spark.eulercircuit.impl;

import DSPPCode.spark.eulercircuit.question.EulerCircuit;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.*;

public class EulerCircuitImpl extends EulerCircuit {

  @Override
  public boolean isEulerCircuit(JavaRDD<String> lines, JavaSparkContext jsc) {
    // Step 1: Parse the input to build the graph
    JavaRDD<Tuple2<Integer, Integer>> edges = lines.map(line -> {
      String[] parts = line.split(" ");
      return new Tuple2<>(Integer.parseInt(parts[0]), Integer.parseInt(parts[1]));
    });

    // Step 2: Build the adjacency list and degree map
    Map<Integer, List<Integer>> adjacencyList = new HashMap<>();
    Map<Integer, Integer> degreeMap = new HashMap<>();

    edges.collect().forEach(edge -> {
      int u = edge._1();
      int v = edge._2();

      adjacencyList.computeIfAbsent(u, k -> new ArrayList<>()).add(v);
      adjacencyList.computeIfAbsent(v, k -> new ArrayList<>()).add(u);

      degreeMap.put(u, degreeMap.getOrDefault(u, 0) + 1);
      degreeMap.put(v, degreeMap.getOrDefault(v, 0) + 1);
    });

    // Step 3: Check if all vertices have even degree
    for (int degree : degreeMap.values()) {
      if (degree % 2 != 0) {
        return false;
      }
    }

    // Step 4: Check if the graph is connected
    // We can use a BFS or DFS from any starting node
    Set<Integer> visited = new HashSet<>();
    Queue<Integer> queue = new LinkedList<>();

    // Start BFS/DFS from the first node in the adjacency list
    Integer startNode = adjacencyList.keySet().iterator().next();
    queue.add(startNode);
    visited.add(startNode);

    while (!queue.isEmpty()) {
      Integer current = queue.poll();
      for (Integer neighbor : adjacencyList.get(current)) {
        if (!visited.contains(neighbor)) {
          visited.add(neighbor);
          queue.add(neighbor);
        }
      }
    }

    // Check if all vertices with at least one edge are visited
    for (Integer vertex : adjacencyList.keySet()) {
      if (!visited.contains(vertex)) {
        return false;
      }
    }

    // If all checks pass, it is an Euler circuit
    return true;
  }
}

