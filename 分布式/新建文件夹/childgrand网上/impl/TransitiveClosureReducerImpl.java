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
  private int num=0;
  @Override
  public void reduce(Text key, Iterable<Text> values,
      Reducer<Text, Text, Text, Text>.Context context) throws IOException, InterruptedException {
    // Set<String> result2 = parent_child.getOrDefault(key.toString(), new HashSet<>());
    // for(Text parent :values){
    //   //key是子，parent是父，
    //     Set<String> result = child_parent.getOrDefault(parent.toString(), new HashSet<>());
    //     for (String grand: result){
    //       context.write(key, new Text(grand));
    //     }
    //
    //     for (String grand: result2){
    //       context.write(new Text(grand),parent);
    //     }
    //
    //     child_parent.computeIfAbsent(key.toString(), k -> new HashSet<>()).add(parent.toString());
    //     parent_child.computeIfAbsent(parent.toString(), k -> new HashSet<>()).add(key.toString());
    //   }
    //获取value-list中value的child
    List<String> grandChild = new ArrayList<>();
    //获取value-list中value的parent
    List<String> grandParent = new ArrayList<>();
    //左表，取出child放入grand_child
    for (Text text : values) {
      String s = text.toString();
      String[] relation = s.split("\\+");
      String relationType = relation[0];
      String childName = relation[1];
      String parentName = relation[2];
      if ("1".equals(relationType)) {grandChild.add(childName);}
      else {
        grandParent.add(parentName);
      }
    }
    //右表，取出parent放入grand_parent
    int grandParentNum = grandParent.size();
    int grandChildNum = grandChild.size();
    if (grandParentNum != 0 && grandChildNum != 0) {
      for (int m = 0; m < grandChildNum; m++) {
        for (int n = 0; n < grandParentNum; n++) {
          //输出结果
          context.write(new Text(grandChild.get(m)), new Text(grandParent.get(n)));
        }
      }
    }








  }
}
