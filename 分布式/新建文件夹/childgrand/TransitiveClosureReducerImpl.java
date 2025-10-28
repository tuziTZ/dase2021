package DSPPCode.mapreduce.transitive_closure.impl;
import DSPPCode.mapreduce.transitive_closure.question.TransitiveClosureMapper;
import DSPPCode.mapreduce.transitive_closure.question.TransitiveClosureReducer;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class TransitiveClosureReducerImpl extends TransitiveClosureReducer{
  private Map<String, Set<String>> child_parent = new HashMap<>();
  private Map<String, Set<String>> parent_child = new HashMap<>();

  @Override
  public void reduce(Text key, Iterable<Text> values,
      Reducer<Text, Text, Text, Text>.Context context) throws IOException, InterruptedException {
    Set<String> result2 = parent_child.getOrDefault(key.toString(), new HashSet<>());
    for(Text parent :values){
      //key是子，parent是父，
        Set<String> result = child_parent.getOrDefault(parent.toString(), new HashSet<>());
        for (String grand: result){
          context.write(key, new Text(grand));
        }

        for (String grand: result2){
          context.write(new Text(grand),parent);
        }

        child_parent.computeIfAbsent(key.toString(), k -> new HashSet<>()).add(parent.toString());
        parent_child.computeIfAbsent(parent.toString(), k -> new HashSet<>()).add(key.toString());
      }





  }
}
