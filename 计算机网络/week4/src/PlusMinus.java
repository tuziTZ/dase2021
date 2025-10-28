public class PlusMinus {
    public int num;
    synchronized public void plusOne(){
        num = num + 1;
    }
//    public void plusOne(){
//        num = num + 1;
//    }
    public void minusOne(){
        num = num - 1;
    }
    public int printNum(){
        return num;
    }
}