public class TestDeadLock {
    public static void main(String[] args) throws InterruptedException {
        PlusMinus plusMinus1 = new PlusMinus();
        plusMinus1.num = 1000;
        PlusMinus plusMinus2 = new PlusMinus();
        plusMinus2.num = 1000;
        PlusMinus plusMinus3 = new PlusMinus();
        plusMinus3.num = 1000;
        MyThread2 thread1 = new MyThread2(plusMinus1, plusMinus2, plusMinus3,1);
        MyThread2 thread2 = new MyThread2(plusMinus1, plusMinus2,plusMinus3, 2);
        MyThread2 thread3 = new MyThread2(plusMinus1, plusMinus2,plusMinus3, 3);
        Thread t1 = new Thread(thread1);
        Thread t2 = new Thread(thread2);
        Thread t3 = new Thread(thread3);
        t1.start();
        t2.start();
        t3.start();
        t1.join();
        t2.join();
        t3.join();
    }
}
class MyThread2 implements Runnable {
    @Override
    public void run() {
        if (tid == 1) {
            synchronized (pm2) {
                System.out.println("thread" + tid + "正在占⽤ plusMinus2");
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("thread" + tid + "试图继续占⽤ plusMinus3");
                System.out.println("thread" + tid + "等待中...");
                synchronized (pm3) {
                    System.out.println("thread" + tid + "成功占⽤了 plusMinus3");
                }
            }
        }else if (tid == 2) {
            synchronized (pm3) {
                System.out.println("thread" + tid + "正在占⽤ plusMinus3");
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("thread" + tid + "试图继续占⽤ plusMinus1");
                System.out.println("thread" + tid + "等待中...");
                synchronized (pm1) {
                    System.out.println("thread" + tid + "成功占⽤了 plusMinus1");
                }
            }
        }
        else if(tid==3){
            synchronized (pm1){
                System.out.println("thread" + tid + "正在占⽤ plusMinus1");
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
                System.out.println("thread" + tid + "试图继续占⽤ plusMinus2");
                System.out.println("thread" + tid + "等待中...");
                synchronized (pm2) {
                    System.out.println("thread" + tid + "成功占⽤了 plusMinus2");
                }
            }
        }
    }
    MyThread2(PlusMinus _pm1, PlusMinus _pm2,PlusMinus _pm3, int _tid) {
        this.pm1 = _pm1;
        this.pm2 = _pm2;
        this.pm3 = _pm3;
        this.tid = _tid;
    }
    PlusMinus pm1;
    PlusMinus pm2;
    PlusMinus pm3;
    int tid;
}
class PlusMinus {
    public int num;
    public void plusOne() {
        num = num + 1;
    }
    public void minusOne() {
        num = num - 1;
    }
    public int printNum() {
        return num;
    }
}