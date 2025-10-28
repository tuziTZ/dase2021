import java.util.Random;

public class Fish implements Animal,Comparable<Fish>{
    public void eat(){
        System.out.println("Fish eats");
    }
    public void travel(){
        System.out.println("Fish travels");
    }
    public void move(){
        System.out.println("Fish moves");
    }

    int size;
    public Fish(){
        Random r = new Random();
        this.size = r.nextInt(100);
    }
    void print(){
        System.out.print(this.size + " < ");
    }
    @Override
    public int compareTo(Fish fish) {
        return this.size-fish.size;
    }
}
