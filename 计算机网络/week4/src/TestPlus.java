public class TestPlus {
    public static void main(String[] args) throws InterruptedException {
        PlusMinus plusMinus = new PlusMinus();
        plusMinus.num = 0;
        int threadNum = 10;
        Thread[] plusThreads = new Thread[threadNum];
        for(int i=0;i<threadNum;i++){
            plusThreads[i] = new Plus(plusMinus);
        }
        for(int i=0;i<threadNum;i++){
            plusThreads[i].start();
//            plusThreads[i].join();
        }
        for(int i=0;i<threadNum;i++){
            plusThreads[i].join();
        }
        System.out.println(plusMinus.printNum());
    }
}
class Plus extends Thread{
    Plus(PlusMinus pm){
        this.plusMinus = pm;
    }
    @Override
    public void run(){
        for(int i=0;i<10000;i++){
//            synchronized (plusMinus){
//                plusMinus.plusOne();
//            }
            plusMinus.plusOne();
        }
    }
    final PlusMinus plusMinus;
}
