
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

//public class test {
//    //1
////    public static void main(String[] args) {
////        StopWatch sw = new StopWatch();
//////        sw.start();
//////        sw.stop();
////        try {
////            sw.start();
////            Thread.sleep(1000);
////            sw.stop();
////        } catch (InterruptedException e) {
////            e.printStackTrace();
////        }
////
////        System.out.println(sw.getElapsedTime());
////
////    }
//    //2
////    public static void main(String[] args){
////        Fish[] myFish=new Fish[10];
////        for(int i=0;i<10;i++){
////            myFish[i]=new Fish();
////        }
////
////        Arrays.sort(myFish);
////        for(int i=0;i<10;i++){
////            myFish[i].print();
////        }
////
////    }
//    //3
////    public static void main(String[] args){
////        SalesEmployee x=new SalesEmployee("AA",2000,0.6f,3);
////        SalariedEmployee y=new SalariedEmployee("BB",3);
////        HourlyEmployee z=new HourlyEmployee("CC",20,24,3);
////        System.out.println(x.getSalary(3)+y.getSalary(3)+z.getSalary(3));
////
////    }
//    //5
////    public static void main(String[] args){
////        try{
////            int[] arr=new int[2];
////            arr[2]=1;
////        }catch(ArrayIndexOutOfBoundsException e){
////            System.out.println("Exception thrown :" + e);
////        }
////        try{
////            Fish fish = null;
////            fish.print();
////        }catch (NullPointerException e){
////            System.out.println("Exception thrown :" + e);
////        }
////        try{
////            int x=3/0;
////        }catch (ArithmeticException e){
////            System.out.println("Exception thrown :" + e);
////        }
////        try{
////            String number ="1234A";
////            Integer.parseInt(number);
////        }catch (NumberFormatException e){
////            System.out.println("Exception thrown :" + e);
////        }
////        try{
////            Employee y=new SalariedEmployee("BB",3);
////            SalesEmployee se=(SalesEmployee) y;
////        }catch (ClassCastException e){
////            System.out.println("Exception thrown :" + e);
////        }
////    }
//
//}
public class test {
    public static void main(String[] args) {
        ArrayList<Color> list = new ArrayList<>();
        for(int i=1;i<=3;i++){
            Collections.addAll(list, Color.values());
        }
        Random r = new Random(1234567);
        Collections.shuffle(list, r);
        for(int i=0;i<list.size();i++){
            Color c = list.get(i);
            switch (c.type){
                case 1:
                    System.out.print("red ");
                    break;
                case 2:
                    System.out.print("green ");
                    break;
                case 3:
                    System.out.print("blue ");
                    break;
            }
        }
    }
}
enum Color{
    RED(1),
    GREEN(2),
    BLUE(3);
    final int type;
    Color(int _type){
        this.type = _type;
    }
}